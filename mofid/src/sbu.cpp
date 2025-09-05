// Main MOFid code to decompose a CIF into nodes, linkers, and solvents.
// Writes the SMILES and catenation info to stdout, errors to stderr, and
// relevant CIFs and simplified topology.cgd to the Output/ directory.

// See https://openbabel.org/docs/dev/UseTheLibrary/CppExamples.html
// Get iterator help from http://openbabel.org/dev-api/group__main.shtml
// Visualize with http://baoilleach.webfactional.com/site_media/blog/emscripten/openbabel/webdepict.html

// Instead of including code to display the full MOF SMILES, just use openbabel natively:
// obabel CIFFILE -ap -ocan

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include <openbabel/mol.h>
#include <openbabel/obiter.h>
#include <openbabel/babelconfig.h>
#include <openbabel/elements.h>

#include "config_sbu.h"
#include "invector.h"
#include "obdetails.h"
#include "deconstructor.h"
#include "framework.h"
#include "periodic.h"
#include "pseudo_atom.h"
#include "topology.h"
#include "virtual_mol.h"


using namespace OpenBabel;  // See http://openbabel.org/dev-api/namespaceOpenBabel.shtml


// Function prototypes
std::string analyzeMOF(std::string filename, const std::string &output_dir=DEFAULT_OUTPUT_PATH);
extern "C" void analyzeMOFc(const char *cifdata, char *analysis, int buflen);
extern "C" int SmilesToSVG(const char* smiles, int options, void* mbuf, unsigned int buflen);
void try_mkdir(const std::string &path);
void write_string(const std::string &contents, const std::string &path);


int main(int argc, char* argv[])
{
	obErrorLog.SetOutputLevel(obWarning);  // See also http://openbabel.org/wiki/Errors

	// Parse args and set up the output directory
	// TODO: consider adding an arg to switch which algorithm is called (MOFid, InChIKey, all-node, etc.)
	char* filename;
	if (argc != 2 && argc != 3) {  // The program name (bin/sbu) also counts as an arg
		std::cerr << "Incorrect number of arguments.  Need to specify the CIF and optionally an output directory." << std::endl;
		return(2);
	}
	filename = argv[1];
	std::string output_dir = DEFAULT_OUTPUT_PATH;
	if (argc >= 3) {
		output_dir = std::string(argv[2]);
	}
	try_mkdir(output_dir);
	try_mkdir(output_dir + METAL_OXO_SUFFIX);
	try_mkdir(output_dir + SINGLE_NODE_SUFFIX);
	try_mkdir(output_dir + ALL_NODE_SUFFIX);
	try_mkdir(output_dir + STANDARD_ISOLATED_SUFFIX);

    // Set up the babel data directory to use a local copy customized for MOFs
	// (instead of system-wide Open Babel data)
	std::stringstream dataMsg;
	dataMsg << "Using local Open Babel data saved in " << LOCAL_OB_DATADIR << std::endl;
	obErrorLog.ThrowError(__FUNCTION__, dataMsg.str(), obAuditMsg);
	dataMsg << "Using local Open Babel shared libraries saved in " << LOCAL_OB_LIBDIR << std::endl;
	obErrorLog.ThrowError(__FUNCTION__, dataMsg.str(), obAuditMsg);
	// Use setenv instead of putenv, per advice about string copies vs. pointers: http://stackoverflow.com/questions/5873029/questions-about-putenv-and-setenv/5876818#5876818
	// This is similar to the approach of cryos/avogadro:main.cpp:127
	// Per my objective, this only sets the environment within the scope of the sbu.exe program.
	// But Windows defines a separate _putenv, etc: https://stackoverflow.com/questions/17258029/c-setenv-undefined-identifier-in-visual-studio
#ifdef _WIN32
	_putenv_s("BABEL_DATADIR", LOCAL_OB_DATADIR);
	_putenv_s("BABEL_LIBDIR", LOCAL_OB_LIBDIR);
#else
	setenv("BABEL_DATADIR", LOCAL_OB_DATADIR, 1);
	setenv("BABEL_LIBDIR", LOCAL_OB_LIBDIR, 1);
#endif

	std::string mof_results = analyzeMOF(std::string(filename), output_dir);
	if (mof_results == "") {  // No MOFs found
		return(1);
	} else {
		std::cout << mof_results;
		return(0);
	}
}

std::string analyzeMOF(std::string filename, const std::string &output_dir) {
	// Extract components of the MOFid
	// Reports nodes/linkers, number of nets found, and writes CIFs to the DEFAULT_OUTPUT_PATH folder

	OBMol orig_mol;
	// Massively improving performance by skipping kekulization of the full MOF
	if (!importCIF(&orig_mol, filename, false)) {
		std::cerr << "Error reading file: %s" << filename << std::endl;
		return "";
	}

	// Save a copy of the original mol for debugging
	writeCIF(&orig_mol, output_dir + "/orig_mol.cif");
	write_string(filename, output_dir + "/mol_name.txt");

	MetalOxoDeconstructor simplifier(&orig_mol);
	std::string metal_oxo_dir = output_dir + METAL_OXO_SUFFIX;
	simplifier.SetOutputDir(metal_oxo_dir);
	simplifier.SimplifyMOF();
	simplifier.WriteCIFs();
	write_string(simplifier.GetMOFkey(), metal_oxo_dir + "/mofkey_no_topology.txt");
	write_string(simplifier.GetLinkerInChIs(), metal_oxo_dir + "/inchi_linkers.txt");
	write_string(simplifier.GetLinkerStats(), metal_oxo_dir + "/linker_stats.txt");

	SingleNodeDeconstructor sn_simplify(&orig_mol);
	sn_simplify.SetOutputDir(output_dir + SINGLE_NODE_SUFFIX);
	sn_simplify.SimplifyMOF();
	sn_simplify.WriteCIFs();

	AllNodeDeconstructor an_simplify(&orig_mol);
	an_simplify.SetOutputDir(output_dir + ALL_NODE_SUFFIX);
	an_simplify.SimplifyMOF();
	an_simplify.WriteCIFs();

	StandardIsolatedDeconstructor std_simplify(&orig_mol);
	std_simplify.SetOutputDir(output_dir + STANDARD_ISOLATED_SUFFIX);
	std_simplify.SimplifyMOF();
	std_simplify.WriteCIFs();

	return simplifier.GetMOFInfo();
}

extern "C" {
void analyzeMOFc(const char *cifdata, char *analysis, int buflen) {
	// Wrap analyzeMOF with C compatibility for Emscripten usage
	// Make the return value an input param, since c_str is (likely?) a pointer to
	// an internal data structure in std::string, which will go out of scope.
	// A good buflen for the output might be 2^16, or 65536
	const char *TEMP_FILE = "from_emscripten.cif";
	std::ofstream cifp(TEMP_FILE, std::ios::out | std::ios::trunc);
	cifp << std::string(cifdata);
	cifp.close();
	COPY_ALL_CIFS_TO_PDB = true;

	strncpy(analysis, analyzeMOF(TEMP_FILE).c_str(), buflen);
}

int SmilesToSVG(const char* smiles, int options, void* mbuf, unsigned int buflen) {
	// From Noel O'Boyle: https://baoilleach.blogspot.com/2015/02/cheminformaticsjs-open-babel.html
	// See also http://baoilleach.webfactional.com/site_media/blog/emscripten/openbabel/webdepict.html
	OpenBabel::OBMol mol;
	OpenBabel::OBConversion conv;
	conv.SetInFormat("smi");

	bool ok = conv.ReadString(&mol, smiles);
	if (!ok) return -1;

	if (options==0) {
		conv.SetOutFormat("ascii");
		conv.AddOption("a", OpenBabel::OBConversion::OUTOPTIONS, "2.2");

	} else {
		conv.SetOutFormat("svg");
		conv.AddOption("C", OpenBabel::OBConversion::OUTOPTIONS);
		conv.AddOption("P", OpenBabel::OBConversion::OUTOPTIONS, "500");
	}

	std::string out = conv.WriteString(&mol);

	if (out.size()+1 >= buflen)
			return -1;

	char* dst = (char*)mbuf;
	strcpy(dst, out.c_str());
	return out.size();
}
}  // extern "C"


void try_mkdir(const std::string &path) {
	// Makes a new directory if it does not exist, raising a warning if it's new
	int created_new_dir = mkdir(path.c_str(), 0755);  // may need _mkdir for Windows
	if (created_new_dir == 0) {
		std::cerr << "Created a new output directory: " << path << std::endl;
	}
}

void write_string(const std::string &contents, const std::string &path) {
	std::ofstream file_info;
	file_info.open(path.c_str(), std::ios::out | std::ios::trunc);
	if (file_info.is_open()) {
		file_info << contents << std::endl;
		file_info.close();
	}
}

