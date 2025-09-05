"""
Calculate the expected MOF components from filename and compare against MOFid

Use the CSD criteria (no bonds to metals) in my modified OpenBabel code and
sbu.cpp to decompose MOFs into fragments.  Compare actual fragments from
ToBACCo or GA hMOF structures against their 'recipe.'

See comments in summarize, GAMOFs/TobaccoMOFs.expected_mofid,
MOFCompare._test_generated, and smiles_diff.py for documentation on the error
classes (and other warnings) reported in the json output from this script.

@author: Ben Bucior
"""

import os
import sys
import glob
import json
import time
import copy
from mofid.cpp_cheminformatics import (ob_normalize, openbabel_replace,
    openbabel_contains)
from mofid.id_constructor import assemble_mofid, parse_mofid
from mofid.smiles_diff import multi_smiles_diff as diff
from mofid.run_mofid import cif2mofid
from mofid.paths import resources_path, mofid_path

# Locations of important files, relative to the Python source code
GA_DB = os.path.join(resources_path,'ga_hmof_info.json')
TOBACCO_DB = os.path.join(resources_path,'tobacco_info.json')
KNOWN_DB = os.path.join(resources_path,'known_mof_info.json')
KNOWN_DEFAULT_CIFS = os.path.join(resources_path,'TestCIFs')
TOBACCO_DEFAULT_CIFS = os.path.join(mofid_path,'Data','tobacco_L_12','quick')
NO_ARG_CIFS = KNOWN_DEFAULT_CIFS  # KnownMOFs() comparisons are used if no args are specified.  See arg parsing of main
PRINT_CURRENT_MOF = True
EXPORT_CODES = True  # Should the read linker/cat/etc. codes from the filename be reported to a '_codes' field in the output JSON?
if sys.version_info[0] < 3:
    py2 = True
else:
    py2 = False

def any(member, list):
    # Is the member any part of the list?
    return member in list

def basename(path):
    # Get the basename for a given path, without the file extension
    return os.path.splitext(os.path.basename(path))[0]

def mof_log(msg):
    # Logging helper function, which writes to stderr to avoid polluting the json
    if PRINT_CURRENT_MOF:
        sys.stderr.write(msg)

def _get_first_atom(smiles):
    # Extract the first atom from a SMILES string, coarsely
    if smiles[0] == '[':
        close_bracket = smiles.find(']')
        return smiles[0:(close_bracket+1)]
    elif smiles[0] in ['c', 'b', 'n', 'o', 's', 'p']:  # Aromatic
        return smiles[0]
    elif not smiles[0].isupper():  # Numbers, parentheses, etc.
        raise ValueError('Unexpected beginning to SMILES string')

    if smiles[1].islower():  # e.g. Cu
        return smiles[0:2]
    else:
        return smiles[0]

def extend_molecule(base, extension, connection_start=20, pseudo_atom='[Lr]'):
    # Extends 'base' SMILES by 'extension' at pseudo atom sites.
    # Implemented through appending dot-separated components, which are linked together
    # through ring IDs.  Starting with %20 will be large enough for most ring systems.
    if pseudo_atom not in base or pseudo_atom not in extension:
        raise ValueError('Pseudo atom not found in SMILES!')
    if connection_start < 10 or connection_start > 90:
        raise ValueError('Invalid connection numbers.  Must be two-digit with buffer!')

    base_parts = base.split(pseudo_atom)
    assert len(base_parts) > 1
    extension_parts = extension.split(pseudo_atom)
    assert len(extension_parts) == 2

    def normalize_parts(parts):
        if parts[0] == '':  # SMILES starts with the pseudo atoms
            first = _get_first_atom(parts[1])
            parts[0] = first  # Attachment point is on the first atom
            parts[1] = parts[1][len(first):]
    normalize_parts(base_parts)
    normalize_parts(extension_parts)

    extended = ''
    connection = connection_start
    while len(base_parts):
        extended += base_parts.pop(0)
        if len(base_parts):  # not the list item in the list
            extended += '%' + str(connection)
            connection += 1

    for link in range(connection_start, connection):  # connection is last used + 1
        extended += '.' + extension_parts[0] + '%' + str(link) + extension_parts[1]

    return extended

def compare_mofids(mofid1, mofid2, names=None):
    # Compares MOFid strings to identify sources of difference, if any
    if names is None:
        names = ['mof1', 'mof2']
    if mofid1 is None or mofid2 is None:
        mof_name = 'Undefined'
        for x in [mofid1, mofid2]:
            if x is not None:
                mof_name = parse_mofid(x)['name']
        return {'match': 'NA',
                'errors': ['Undefined composition'],
                'topology': None,
                'smiles': None,
                'cat': None,
                names[0]: mofid1,
                names[1]: mofid2,
                'name': mof_name
                }
    parsed = [parse_mofid(x) for x in [mofid1, mofid2]]
    comparison = dict()
    comparison['match'] = True
    comparison['errors'] = []
    comparison[names[0]] = mofid1
    comparison[names[1]] = mofid2
    for key in parsed[0]:
        expected = parsed[0][key]
        if parsed[1][key] == expected:
            comparison[key] = expected
            continue
        elif key == 'topology':  # Handling multiple, alternate topological definitions
            other_topologies = parsed[1][key].split(',')
            matched_topology = False
            for topology in other_topologies:
                if topology == expected:  # If any of them match
                    comparison[key] = topology
                    matched_topology = True
            if matched_topology:
                continue
        # Else, it's a mismatch, so report an error as 'err_<KEY TYPE>',
        # e.g. 'err_topology'
        comparison[key] = False
        comparison['match'] = False
        comparison['errors'].append('err_' + key)

    # Deeper investigation of SMILES-type errors
    if 'err_smiles' in comparison['errors']:
        comparison['errors'].remove('err_smiles')
        for err in diff(parsed[0]['smiles'], parsed[1]['smiles']):
            comparison['errors'].append('err_' + err)

    return comparison

def summarize(results):
    # Summarize the error classes for MOFid results.
    # 'error_types' from the header of the .json output are defined below
    # in the main for loop.
    summarized = {'mofs': results, 'errors': dict()}
    error_types = {'err_topology': 0, 'err_smiles': 0, 'two': 0, 'three_plus': 0, 'success': 0, 'undefined': 0}
    for match in results:
        if match['match'] == 'NA':  # NA cases where we don't know the nodes and linkers
            assert 'Undefined composition' in match['errors']
            error_types['undefined'] += 1
        elif match['match']:
            error_types['success'] += 1
        elif len(match['errors']) == 2:
            error_types['two'] += 1
        elif len(match['errors']) > 2:
            error_types['three_plus'] += 1
        elif len(match['errors']) == 1:
            # Other classes of known issue with MOFid generation and/or naming
            # scheme are copied verbatim, e.g. 'err_'* from the
            # `compare_mofids` function directly before this one.
            known_issue = match['errors'][0]
            if known_issue not in error_types:
                error_types[known_issue] = 0  # Initialize new class of errors
            error_types[known_issue] += 1
        else:
            if 'unreliable_summary' not in error_types:
                error_types['unreliable_summary'] = 0
            error_types['unreliable_summary'] += 1

    summarized['errors']['error_types'] = error_types
    summarized['errors']['total_cifs'] = len(results)
    summarized['errors']['elapsed_time'] = sum([mof['time'] for mof in results])
    return summarized

class MOFCompare:
    # Compares a MOF's decomposition against its expected components
    def __init__(self):
        self.db_file = None
        # self.load_components()  # Used in subclasses
        raise UserWarning('MOF parsing only (partially) implemented for GA hMOFs and ToBACCo')

    def load_components(self, db_file = None):
        # Load node and linker compositions from a saved JSON definition (to keep this file cleaner)
        if db_file is None:
            db_file = self.db_file
        with open(db_file, 'r') as inp:
            mof_db = json.load(inp)
        self.mof_db = mof_db
        return None

    def extract_db_smiles(self, db_dict):
        # Save all the nodes and linker definitions to the current directory
        # Then, convert to a big svg with
        # obabel linkers.can -O linkers.svg -xe -xC
        # Note: this function and the json file will be deprecated by the metal+linker
        # split paradigm suggested at group meeting ('molecule subtraction')
        linkers = db_dict['linkers']
        for id in linkers:
            print(linkers[id] + ' ' + id)

    def transform_mofid(self, mofid):
        # Transforms the raw MOFid read into a script.
        # Override in subclasses, if applicable.
        # Used by the GA hMOFs to get the skeleton of the functionalized MOFs, and only compare the unfunctionalized versions
        return mofid

    def test_auto(self, spec):
        # Automatically test a MOF spec, given either as a CIF filename or MOFid string
        if 'MOFid' in spec:
            return self.test_mofid(spec)
        elif os.path.exists(spec):
            return self.test_cif(spec)
        else:
            raise ValueError('Unknown specification for testing: ' + spec)

    def test_mofid(self, mofid):
        # Test a generated MOFid string against the expectation based on the CIF filename
        start = time.time()
        cif_path = parse_mofid(mofid)['name']
        return self._test_generated(cif_path, mofid, start, 'from_mofid')

    def test_cif(self, cif_path):
        # Compares an arbitrary CIF file against its expected specification
        # Returns a formatted JSON string with the result
        start = time.time()
        auto_ids = cif2mofid(cif_path=cif_path)
        return self._test_generated(cif_path, auto_ids['mofid'],
            start, 'from_cif', auto_ids['mofkey'])

    def _test_generated(self,
        cif_path, generated_mofid,
        start_time = None, generation_type = 'from_generated',
        mofkey = None
        ):
        # Compares an arbitrary MOFid string against the value generated,
        # either locally in the script (cif2mofid) or from an external .smi file.
        # Also tests for common classes of error
        mofid_from_name = self.expected_mofid(cif_path)

        if mofid_from_name is None:  # missing SBU info in the DB file
            return None  # Currently, skip reporting of structures with undefined nodes/linkers

        if (py2 and type(mofid_from_name) in [str, unicode]) or (not py2 and type(mofid_from_name) is str):
            orig_mofid = mofid_from_name
            mofid_from_name = dict()
            mofid_from_name['default'] = orig_mofid
        default = parse_mofid(mofid_from_name['default'])
        fragments = default['smiles'].split('.')

        # Define sources of error when the program exits with errors.
        # Without these definitions, the validator would return a generic
        # class of error, e.g. 'err_topology', instead of actually indicating
        # the root cause from program error or timeout.
        mofid_from_name['err_timeout'] = assemble_mofid(
            fragments, 'TIMEOUT', default['cat'], mof_name=default['name'])
        mofid_from_name['err_systre_error'] = assemble_mofid(
            fragments, 'ERROR', default['cat'], mof_name=default['name'])
        mofid_from_name['err_cpp_error'] = assemble_mofid(
            ['*'], 'NA', None, mof_name=default['name'])
        mofid_from_name['err_no_mof'] = assemble_mofid(
            ['*'], 'NA', 'no_mof', mof_name=default['name'])


        # Run transformations on the generated MOFid from CIF or smi database, if applicable (e.g. GA hMOFs)
        test_mofid = self.transform_mofid(generated_mofid)
        if test_mofid != generated_mofid:
            generation_type += '_transformed'

        if test_mofid is None and generated_mofid is not None:
            comparison = self.compare_multi_mofid(mofid_from_name,
                generated_mofid, ['from_name', generation_type])
            comparison['errors'] = ['err_missing_transform']
            comparison['match'] = False
        else:
            # Calculate the MOFid derived from the CIF structure itself
            comparison = self.compare_multi_mofid(mofid_from_name,
                test_mofid, ['from_name', generation_type])

        if start_time is None:
            comparison['time'] = 0
        else:
            comparison['time'] = time.time() - start_time
        if mofkey is not None:
            comparison['mofkey_from_cif'] = mofkey
        comparison['name_parser'] = self.__class__.__name__
        return comparison

    def compare_multi_mofid(self, multi_mofid1, mofid2, names=None):
        # Allow multiple reference MOFids for comparison against the extracted version
        # to account for known issues in either the naming or reference material (ambiguities, etc)

        if (py2 and type(multi_mofid1) in [str, unicode]) or (not py2 and type(multi_mofid1) is str):
            return compare_mofids(multi_mofid1, mofid2, names)

        assert type(multi_mofid1) == dict  # Else, let's handle multiple references
        assert 'default' in multi_mofid1
        default_comparison = compare_mofids(multi_mofid1['default'],
            mofid2, names)
        if EXPORT_CODES and '_codes' in multi_mofid1:
            default_comparison['_codes'] = multi_mofid1['_codes']
        if default_comparison['match']:
            return default_comparison

        for test in multi_mofid1.keys():
            if test in ['default', '_codes']:
                continue
            test_mofid = multi_mofid1[test]
            test_comparison = compare_mofids(test_mofid, mofid2, names)
            if test_comparison['match']:
                test_comparison['match'] = False  # Should it be reported as a match if we know the source of error?
                test_comparison['errors'] = [test]
                if EXPORT_CODES and '_codes' in multi_mofid1:
                    test_comparison['_codes'] = multi_mofid1['_codes']
                return test_comparison

        return default_comparison  # No special cases apply



class KnownMOFs(MOFCompare):
    # Minimal class which doesn't have to do much work to scour the database of known MOFs.
    # Excellent as a integration test for my code, i.e. did my changes cause anything else to obviously break?
    def __init__(self):
        self.db_file = KNOWN_DB
        self.load_components()

    def parse_filename(self, mof_path):
        # Extract basename of the MOF.  expected_mofid will convert it to the reference MOFid string
        return basename(mof_path)

    def expected_mofid(self, cif_path):
        # What is the expected MOFid based on the information in a MOF's filename?
        mof_name = self.parse_filename(cif_path)
        if mof_name in self.mof_db:
            return self.mof_db[mof_name]
        else:
            return None



class GAMOFs(MOFCompare):
    # Gene-based reduced WLLFHS hMOF database in Greg and Diego's 2016 paper
    # Ref: https://doi.org/10.1126/sciadv.1600909
    def __init__(self):
        self.db_file = GA_DB
        self.load_components()

    def parse_filename(self, hmof_path):
        # Extract hMOF recipes from the filename, formatted as xxxhypotheticalMOF_####_#_#_#_#_#_#.cif
        codes = dict()
        mof_name = basename(hmof_path)  # Get the basename without file extension

        parts = mof_name.split('_')
        assert len(parts) == 8  # hypoMOF + number + 6 chromosomes
        codes['name'] = mof_name
        codes['num'] = parts[1]
        codes['max_cat'] = parts[2]
        codes['cat'] = parts[3]
        codes['nodes'] = parts[4]
        codes['linker1'] = parts[5]
        codes['linker2'] = parts[6]
        codes['functionalization'] = parts[7]

        return codes

    def transform_mofid(self, mofid):
        # De-functionalize MOFids read from CIFs or a database.
        # This will allow easy comparison between the linker skeletons found and expected.
        if mofid is None:
            return None

        cif_path = parse_mofid(mofid)['name']
        codes = self.parse_filename(cif_path)
        fg = codes['functionalization']

        if fg == '0':
            return mofid
        if fg not in self.mof_db['functionalization']:
            return None  # Raises a transform error

        fragments = mofid.split()[0]
        fancy_name = ' '.join(mofid.split()[1:])
        pattern = self.mof_db['functionalization'][fg]
        if not openbabel_contains(fragments, pattern):
            return None  # will raise a transform error in the output

        skeletons = [openbabel_replace(x, pattern, '[#1:1]') for x in fragments.split('.')]
        skeletons = '.'.join(skeletons).split('.')  # Handle transformations that split apart building blocks into multiple parts
        skeletons = list(set(skeletons))  # Only keep unique backbones if they have different functionalization patterns
        if '' in skeletons:  # null linker from defunctionalization on a lone functional group
            skeletons.remove('')
        skeletons.sort()

        return ' '.join(['.'.join(skeletons), fancy_name])  # Reconstruct the defunctionalized MOFid

    def expected_mofid(self, cif_path):
        # What is the expected MOFid based on the information in a MOF's filename?
        codes = self.parse_filename(cif_path)
        code_key = {
            'nodes': 'nodes',
            'linker1': 'linkers',
            'linker2': 'linkers',
            'functionalization': 'functionalization',
            # Catenation is handled below, with the topology statement
        }

        is_component_defined = []
        for key in code_key:
            #is_component_defined.extend([x in self.mof_db[code_key[key]] for x in codes[key]])
            is_component_defined.extend([codes[key] in self.mof_db[code_key[key]]])

        topology = self._topology_from_gene(codes)
        codes['default_gene_topology'] = topology
        cat = codes['cat']

        if not any(False, is_component_defined):  # Everything is defined.  Why didn't I use Python's built-in `all`?  Test this later.
            sbus = []  # more rigorously, MOF building blocks
            sbu_codes = ['nodes', 'linker1', 'linker2']  # Functionalization handled in transform_mofid
            n_components = []  # Potentially inconsistent ordering of paddlewheel pillars
            n_orig = []  # Pillar molecule before transformation
            for part in sbu_codes:
                full_smiles = self.mof_db[code_key[part]][codes[part]].split('.')
                for smiles in full_smiles:
                    if smiles not in sbus:
                        sbus.append(smiles)
                # Also generate the nitrogen-terminated versions for pillared paddlewheels
                if (part in ['linker1', 'linker2'] and
                    codes['nodes'] in ['1', '2'] and topology == 'pcu'
                    ):
                    assert len(full_smiles) == 1
                    n_smi = self._carboxylate_to_nitrogen(full_smiles[0])
                    if n_smi not in n_components:
                        n_components.append(n_smi)
                        n_orig.append(full_smiles[0])
            if len(n_components) == 1:
                sbus.append(n_components[0])
            sbus.sort()

            mofid_options = dict()
            mofid_options['default'] = assemble_mofid(
                sbus, topology, cat, mof_name=codes['name'])

            # Flagging common cases when the generated MOF does not match the idealized recipe,
            # for example when the bottom-up algorithm is missing a connection.
            if topology == 'fcu':  # Zr nodes do not always form **fcu** topology, even when linker1==linker2
                # Note: if we change the handling of bound modulators, we will have to add
                # benzoic acid to these **pcu** MOF nodes
                mofid_options['Zr_mof_not_fcu'] = assemble_mofid(
                    sbus, 'pcu', cat, mof_name=codes['name'])
            if codes['nodes'] == '4':  # Some Zr hMOFs have four linkers replaced by benzoic acid
                mofid_options['Zr_mof_as_hex'] = assemble_mofid(
                    sbus, 'hex', cat, mof_name=codes['name'])
                if topology == 'pcu':
                    mofid_options['Zr_mof_pcu_as_fcu'] = assemble_mofid(
                        sbus, 'fcu', cat, mof_name=codes['name'])
            if topology == 'rna':  # Some large V nodes are geometrically disconnected
                v_sbus = copy.deepcopy(sbus)
                v_sbus.append('[O-]C(=O)c1ccccc1')
                v_sbus.sort()
                mofid_options['V_incomplete_linker'] = assemble_mofid(
                    v_sbus, 'ERROR', cat, mof_name=codes['name'])
            if len(n_components) == 2:
                # Ambiguities arise when there is both a linker1 and linker2
                # because of the combinations between N/carboxylate and L1/L2
                # in pcu paddlewheel MOFs with pillars.
                for i, smi in enumerate(n_components):
                    # Both L1 and L2, plus N-pillar i (unknown which one beforehand)
                    n_sbus = copy.deepcopy(sbus)
                    n_sbus.append(smi)
                    n_sbus.sort()
                    mofid_options['unk_pillar' + str(i+1)] = assemble_mofid(
                        n_sbus, 'pcu', cat, mof_name=codes['name'])

                    # Replacing linker i from a carboxylate to a pillar
                    n_sbus.remove(n_orig[i])
                    n_sbus.sort()
                    mofid_options['replaced_pillar' + str(i+1)] = assemble_mofid(
                        n_sbus, 'pcu', cat, mof_name=codes['name'])

            if EXPORT_CODES:
                mofid_options['_codes'] = codes

            return mofid_options
        else:
            return None

    def _topology_from_gene(self, genes):
        # Expected topology based on the gene-based identification criteria in Table S1 of 10.1126/sciadv.1600909
        zn4o = 0
        paddlewheels = [1, 2]
        v_node = 3
        zr_node = 4
        tritopic = range(30, 34)  # (30-33)
        tetrahedral4 = range(34, 38)  # tetrahedral tetratopic linkers
        planar4 = [38, 39]  # planar tetratopic linkers

        nodes = int(genes['nodes'])
        linker1 = int(genes['linker1'])
        linker2 = int(genes['linker2'])

        if nodes == zn4o:  # Zn4O nodes
            return 'pcu'
        elif nodes in paddlewheels and linker1 < 30 and linker2 < 30:  # Paddlewheels with ditopic linkers
            return 'pcu'
        elif nodes == zr_node and linker1 != linker2:  # Zr nodes and two different linkers
            return 'pcu'

        elif nodes == v_node:  # Vanadium nodes
            return 'rna'  # **sra** in the table, but the **rna** representation is more consistent with the ID scheme
        elif nodes == zr_node and linker1 == linker2:  # Zr nodes and one type of linker
            return 'fcu'

        elif linker1 in tritopic and linker2 in tritopic:
            return 'tbo'
        elif linker1 in tetrahedral4 and linker2 in tetrahedral4:
            return 'dia'
        elif linker1 in planar4 and linker2 in planar4:
            return 'nbo'

        else:
            raise ValueError('Undefined topology for ' + genes['name'])
            return 'UNK'

    def _carboxylate_to_nitrogen(self, linker_smiles):
        # Transforms carboxylate linkers to their nitrogen-terminated versions
        # Implement by deleting the carboxylate and transforming C->N
        if linker_smiles in ['[O-]C(=O)C(=O)[O-]', '[O-]C(=O)C#CC(=O)[O-]']:
            return 'N#N'
        nitrogen_linker = openbabel_replace(linker_smiles, '[#6:1][C:2](=[O:3])[O:4]', '[#7:1]')
        # Open Babel has trouble round-tripping the aromatic nitrogen ring
        # n1cc2ccc3c4c2c(c1)ccc4cnc3 or its rdkit equivalent c1cc2cncc3ccc4cncc1c4c23.
        # Both of these return '[nH]1cc2ccc3c4c2c(c1)ccc4c[nH]c3'
        # So, for these cases and their functionalizations, etc., give kekulization a hand.
        # Note: this bug may be temporary and fixed by new community efforts on kekulization and implicit H.
        nitrogen_linker = nitrogen_linker.replace('[nH]', 'n')
        return nitrogen_linker


class TobaccoMOFs(MOFCompare):
    def __init__(self):
        self.db_file = TOBACCO_DB
        self.load_components()

    def parse_filename(self, tobacco_path):
        # Extract ToBACCo recipes from the filename
        # Format: topology_sym_x_node_type_sym_x_node2_type_L_linkernum.cif ('_' for empty linker)
        # Can we parse this using ToBACCo's own code??
        codes = {'name': None, 'nodes': [], 'linker': None, 'topology': None}
        mof_info = basename(tobacco_path)  # Get the basename without file extension
        codes['name'] = mof_info

        parsed = mof_info.split('_', 1)
        codes['topology'] = parsed[0]
        mof_info = parsed[1]

        while 'sym_' in mof_info:
            parsed = mof_info.split('_')
            codes['nodes'].append('_'.join(parsed[0:4]))
            mof_info = '_'.join(parsed[4:])

        # Not sure why bcs, etc., have an extra underscore in the topology.
        mof_info = mof_info.strip('_')
        if mof_info == '':
            mof_info = 'L__'
        codes['linker'] = mof_info

        return codes

    def expected_mofid(self, cif_path):
        # What is the expected MOFid based on the information in a MOF's filename?
        codes = self.parse_filename(cif_path)
        # Currently skipping B-containing sym_13_mc_12 and sym_16_mc_6
        # sym_24_mc_13 will be incompatible with our current decomposition scheme
        # sym_3_mc_0 is currently causing segfaults, so diagnose this node
        # Also skipping a few more nodes until consistency issues are figured out.

        if (not any(False, [x in self.mof_db['nodes'] for x in 
            codes['nodes']])) and (codes['linker'] in self.mof_db['linkers']):
            # Skip structures with tricky nodes (undefined in the table for now)
            # Apply 'sticky ends' to node/linker definitions
            assert len(codes['nodes']) in [1,2]
            node1 = self.mof_db['nodes'][codes['nodes'][0]]
            if len(codes['nodes']) == 1:
                node2 = self.mof_db['nodes'][codes['nodes'][0]]
            else:
                node2 = self.mof_db['nodes'][codes['nodes'][1]]
            linker = self.mof_db['linkers'][codes['linker']]
            fragments = self.assemble_smiles(node1, node2, linker)
            fragments.sort()

            topology = codes['topology']
            if topology.startswith('test.'):
                topology = topology[5:]
            if len(topology) == 4 and topology.endswith('b'):
                topology = topology[0:3]  # Remove binary designation
            cat = '0'  # All ToBaCCo MOFs are uncatenated

            mofid_options = dict()

            # Generate a reference MOFid based on SBU composition
            mofid_options['default'] = assemble_mofid(
                fragments, topology, cat, mof_name=codes['name'])
            # Known classes of issues go here
            if topology == 'tpt':  # Systre analysis finds an **stp** net for ToBaCCo MOFs with the **tpt** template
                mofid_options['stp_from_tpt'] = assemble_mofid(
                    fragments, 'stp', cat, mof_name=codes['name'])

            if EXPORT_CODES:
                mofid_options['_codes'] = codes

            return mofid_options
        else:
            return None

    def assemble_smiles(self, node1, node2, linker):
        # Assemble the expected nodes and linkers based on the designated compositions in the database,
        # plus transformations to join 'sticky ends' together (using an [Lr] pseudo atom).
        # Returns a list of the SMILES components
        smiles = []
        sticky_ends = []
        for node in [node1, node2]:
            subnodes = node.split('.')
            for part in subnodes:
                if '[Lr]' in part:
                    sticky_ends.append(part)
                else:
                    if part not in smiles:
                        smiles.append(part)

        if linker.count('[Lr]') != 2:
            raise ValueError('Linker must contain two sticky ends')
        if len(sticky_ends) != 2:
            raise ValueError('Both nodes must contain sticky end(s)')
        # Ensure the second component only has to react once in both transformations (node1 + linker, intermediate + node2)
        if sticky_ends[1].count('[Lr]') > 1:
            sticky_ends = [sticky_ends[1], sticky_ends[0]]
        if sticky_ends[1].count('[Lr]') != 1:
            raise ValueError('Both nodes cannot contain multiple sticky ends')

        # React all of the sticky ends.  Temporarily make the linker reactive only on one end, then extend in a second step.
        mod_linker = linker.replace('[Lr]', '[No]').replace('[No]', '[Lr]', 1)  # sub the second [Lr] with [No]
        intermediate = ob_normalize(extend_molecule(sticky_ends[0], mod_linker))
        intermediate = intermediate.replace('[No]', '[Lr]')  # 'Reactivate' the second sticky end of the linker
        organic = ob_normalize(extend_molecule(intermediate, sticky_ends[1]))

        smiles.append(organic)
        smiles.sort()
        return smiles


class AutoCompare:
    # Automatically selects the appropriate MOF comparison class using filename-based heuristics
    # Note: this is not a full implementation of MOFCompare--just a wrapper for test_cif
    # TODO: allow the user to manually specify the MOF class using a command line flag
    def __init__(self, recalculate=False):
        self.known = KnownMOFs()
        self.precalculated = self.known.mof_db.keys()
        self.ga = GAMOFs()
        self.tobacco = TobaccoMOFs()
        # Maybe also a NullMOFs class eventually, which is just a calculator sans comparisons?
        self.recalculate = recalculate  # Should MOFid be recalculated even if known?

    def test_cif(self, cif_path):
        # Dispatch to the class corresponding to the source of the input CIF
        mof_info = basename(cif_path)
        parser = self._choose_parser(mof_info)
        if parser is None:
            return None
        return parser.test_cif(cif_path)

    def test_mofid(self, mofid):
        assert 'MOFid' in mofid
        parser = self._choose_parser(parse_mofid(mofid)['name'])
        if parser is None:
            return None
        return parser.test_mofid(mofid)

    def test_auto(self, spec):
        # Automatically test a MOF spec, given either as a CIF filename or MOFid string
        # See also the implementation in MOFCompare
        if 'MOFid' in spec:
            return self.test_mofid(spec)
        elif os.path.exists(spec):
            return self.test_cif(spec)
        else:
            raise ValueError('Unknown specification for testing: ' + spec)

    def _choose_parser(self, cif_name):
        # Determine the appropriate MOFCompare class based on a MOF's name
        if (not self.recalculate) and (cif_name in self.precalculated):
            mof_log('...using precompiled table of known MOFs\n')
            return self.known
        elif 'hypotheticalmof_' in cif_name.lower() or 'hmof_' in cif_name.lower():
            # Underscore suffix prevents false positives in structures named optimized_hmof1.cif, etc.
            if '_i_' in cif_name.lower():
                raise ValueError('Wilmer 2012 hypothetical MOF format no longer supported.  See updated GA format to run validations.')
            else:
                mof_log('...parsing file with rules for GA hypothetical MOFs\n')
                return self.ga
        elif '_sym_' in cif_name:
            mof_log('...parsing file with rules for ToBACCo MOFs\n')
            return self.tobacco
        else:
            mof_log('...unable to find a suitable rule automatically\n')
            return None


if __name__ == '__main__':
    comparer = AutoCompare()  # By default, guess the MOF type by filename
    input_type = 'CIF'
    args = sys.argv[1:]
    if len(args) == 0:  # validation testing against reference MOFs
        inputs = glob.glob(NO_ARG_CIFS + '/*.[Cc][Ii][Ff]')
        comparer = KnownMOFs()
    elif len(args) == 1 and args[0].endswith('/'):
        # Run a whole directory if specified as a single argument with an ending slash
        inputs = glob.glob(args[0] + '*.[Cc][Ii][Ff]')
        comparer = AutoCompare(True)  # Do not use database of known MOFs
    elif len(args) == 1 and (
        args[0].endswith('.txt') or
        args[0].endswith('.smi') or
        args[0].endswith('.out')
        ):
        input_type = 'MOFid line'
        with open(args[0], 'r') as f:
            inputs = f.readlines()
            inputs = [x.rstrip('\n') for x in inputs]
    else:
        input_type = 'Auto'
        inputs = args

    mofid_results = []
    for num_cif, curr_input in enumerate(inputs):
        display_input = curr_input
        if input_type == 'MOFid line':
            if len(curr_input.split(';')) == 1:
                display_input = curr_input
            else:
                display_input = ';'.join(curr_input.split(';')[1:])
        mof_log(' '.join(['Found', input_type, str(num_cif+1), 'of',
            str(len(inputs)), ':', display_input]) + '\n')
        if input_type == 'CIF':
            result = comparer.test_cif(curr_input)
        elif input_type == 'MOFid line':
            result = comparer.test_mofid(curr_input)
        else:
            result = comparer.test_auto(curr_input)
        if result is not None:
            mofid_results.append(result)

    results_summary = summarize(mofid_results)
    json.dump(results_summary, sys.stdout, indent=4)
    num_mofs = results_summary['errors']['total_cifs']
    num_errors = num_mofs - results_summary['errors']['error_types']['success']
    mof_log(' '.join(['\nResults:', str(num_errors), 'errors in', str(num_mofs), 'MOFs\n']))
    sys.exit(num_errors)
