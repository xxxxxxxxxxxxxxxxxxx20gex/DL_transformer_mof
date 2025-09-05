/* searchdb: searches a .smi DB for a given SMARTS pattern
 * Usage: searchdb '[Co]' core.smi exclude
 * The optional last parameter will reverse the search.
 * Returns the matching SMILES lines
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/babelconfig.h>
#include "config_sbu.h"
#include <cstring>


using namespace OpenBabel;  // See http://openbabel.org/dev-api/namespaceOpenBabel.shtml


// Function prototypes
std::string runSearch(std::string pattern, std::string db_file, bool reverse_search);
extern "C" void runSearchc(const char *pattern, const char *db_file, bool reverse_search, char *out_file);


int main(int argc, char* argv[])
{
	char* pattern = argv[1];
	char* db_file = argv[2];

	bool exclusion_search = false;
	if (argc > 3 && std::strcmp(argv[3], "exclude") == 0) {
		exclusion_search = true;
	}

	// Set up the babel data directory to use a local copy customized for MOFs
	// (instead of system-wide Open Babel data) for this particular program
	setenv("BABEL_DATADIR", LOCAL_OB_DATADIR, 1);
    // Set up the babel shared libraries
    setenv("BABEL_LIBDIR", LOCAL_OB_LIBDIR, 1);

	std::cout << runSearch(pattern, db_file, exclusion_search);
}

std::string runSearch(std::string pattern, std::string db_file, bool reverse_search) {
	OBConversion obconv;
	obconv.SetInFormat("smi");
	obconv.SetOutFormat("copy");
	// WARNING: the "copy" output format requires non-Windows line endings per http://openbabel.org/docs/current/FileFormats/Copy_raw_text.html

	std::string search_type;
	if (reverse_search) {
		search_type = "v";  // inverse search, like `grep -v`
	} else {
		search_type = "s";  // standard SMARTS search
	}

	// Template based on call to obabel:DoOption from the "single character general option"
	// Overall, this code later activates DoTransformations,
	// specifically ops/opisomorph.cpp to perform the SMARTS filtering
	obconv.AddOption(search_type.c_str(), OBConversion::GENOPTIONS, pattern.c_str());

	std::stringstream out_smi;
	std::ifstream infile;
	infile.open(db_file.c_str());
	obconv.Convert(&infile, &out_smi);
	infile.close();

	return out_smi.str();
}

extern "C"
void runSearchc(const char *pattern, const char *db_file, bool reverse_search, char *out_file) {
	// Wrap runSearch for Emscripten
	// Runs a search for pattern on db_file, writing the results to out_file.
	std::ofstream smip(out_file, std::ios::out | std::ios::trunc);
	smip << runSearch(pattern, db_file, reverse_search);
	smip.close();
}  // extern "C"
