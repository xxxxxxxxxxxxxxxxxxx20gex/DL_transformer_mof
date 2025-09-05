/* sobgrep: like obgrep, but searches singly-bonded CIFs for SMARTS patterns */
/* Outputs the CIF name to stdout and coordinates to results.cif if found */

/* New code to analyze bond distances.
 * Modified from src/sbu.cpp, commit 05f5307ff346f2d1768b93752d836fb18270aa1e
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <openbabel/obconversion.h>
#include <openbabel/generic.h>
#include <openbabel/atom.h>
#include <openbabel/mol.h>
#include <openbabel/obiter.h>
#include <openbabel/babelconfig.h>
#include <openbabel/elements.h>
#include <openbabel/parsmart.h>
#include "config_sbu.h"


using namespace OpenBabel;  // See http://openbabel.org/dev-api/namespaceOpenBabel.shtml


// Function prototypes
void local_copyAtom(OBAtom* src, OBMol* dest);
OBMol local_initMOF(OBMol *orig_in_uc);
OBUnitCell* local_getPeriodicLattice(OBMol *mol);


int main(int argc, char* argv[])
{
	obErrorLog.SetOutputLevel(obInfo);  // See also http://openbabel.org/wiki/Errors
	char* filename = argv[1];
	char* pattern = argv[2];

    // Set up the babel data directory to use a local copy customized for MOFs
	// (instead of system-wide Open Babel data)
	std::stringstream dataMsg;
	dataMsg << "Using local Open Babel data saved in " << LOCAL_OB_DATADIR << std::endl;
	obErrorLog.ThrowError(__FUNCTION__, dataMsg.str(), obAuditMsg);
	dataMsg << "Using local Open Babel shared libraries saved in " << LOCAL_OB_LIBDIR << std::endl;
	obErrorLog.ThrowError(__FUNCTION__, dataMsg.str(), obAuditMsg);
#ifdef _WIN32
	_putenv_s("BABEL_DATADIR", LOCAL_OB_DATADIR);
	_putenv_s("BABEL_LIBDIR", LOCAL_OB_LIBDIR);
#else
	setenv("BABEL_DATADIR", LOCAL_OB_DATADIR, 1);
	setenv("BABEL_LIBDIR", LOCAL_OB_LIBDIR, 1);
#endif

	// Read CIF as single bonds
	OBMol orig_mol;
	OBConversion obconversion;
	obconversion.SetInFormat("mmcif");
	obconversion.AddOption("p", OBConversion::INOPTIONS);
	obconversion.AddOption("s", OBConversion::INOPTIONS);
	if (!obconversion.ReadFile(&orig_mol, filename)) {
		printf("Error reading file: %s", filename);
		exit(1);
	}

	// Strip all of the original CIF labels, so they don't interfere with the automatically generated labels in the output
	FOR_ATOMS_OF_MOL(a, orig_mol) {
		if (a->HasData("_atom_site_label")) {
			a->DeleteData("_atom_site_label");
		}
	}

	int num_matches = 0;
	std::set<OBAtom*> matched_atoms;
	// Using example code from: http://openbabel.org/dev-api/classOpenBabel_1_1OBSmartsPattern.shtml
	OBSmartsPattern sp;
	sp.Init(pattern);  // Specified pattern in the args
	sp.Match(orig_mol);
	std::vector<std::vector<int> > maplist;
	maplist = sp.GetUMapList();
	std::vector<std::vector<int> >::iterator i;
	std::vector<int>::iterator j;
	for (i=maplist.begin(); i!=maplist.end(); ++i) {  // Begin cluster
		++num_matches;
		for (j=i->begin(); j!=i->end(); ++j) {  // Within cluster
			matched_atoms.insert(orig_mol.GetAtom(*j));
		}
	}

	OBMol match = local_initMOF(&orig_mol);
	match.BeginModify();
	for (std::set<OBAtom*>::iterator it=matched_atoms.begin(); it!=matched_atoms.end(); ++it) {
		local_copyAtom(*it, &match);
	}
	match.EndModify();
	match.ConnectTheDots();

	if (num_matches) {
		// print number of atoms and number of matches
		std::cout << "# Found " << num_matches;
		std::cout << " matches containing " << match.NumAtoms();
		std::cout << " atoms." << std::endl;
		std::cout << "# Original file:\t" << filename << std::endl;
		std::cout << "# Search pattern:\t" << pattern << std::endl;

		OBConversion cifout;
		cifout.SetOutFormat("cif");
		cifout.AddOption("g");
		std::cout << cifout.WriteString(&match);

	}
	return(0);
}

void local_copyAtom(OBAtom* src, OBMol* dest) {
	// Copies properties of an atom from source to dest,
	// without triggering hybridization, bond order detection, etc.
	OBAtom* copied = dest->NewAtom();
	copied->SetVector(src->GetVector());
	copied->SetAtomicNum(src->GetAtomicNum());
	copied->SetType(OBElements::GetName(src->GetAtomicNum()));
}

OBMol local_initMOF(OBMol *orig_in_uc) {
	// Initializes a MOF with the same lattice params as *orig_in_uc
	OBMol dest;
	dest.SetData(local_getPeriodicLattice(orig_in_uc)->Clone(NULL));
	dest.SetPeriodicMol();
	return dest;
}

OBUnitCell* local_getPeriodicLattice(OBMol *mol) {
	return (OBUnitCell*)mol->GetData(OBGenericDataType::UnitCell);
}
