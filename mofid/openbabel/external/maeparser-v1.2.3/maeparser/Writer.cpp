#include "Writer.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>

#include <fstream>
#include <sstream>
#include <iostream>

#include "MaeBlock.hpp"

using namespace std;
using boost::algorithm::ends_with;
using boost::iostreams::filtering_ostream;
using boost::iostreams::file_sink;

namespace schrodinger
{
namespace mae
{

Writer::Writer(std::shared_ptr<ostream> stream) : m_out(stream)
{
    write_opening_block();
}

Writer::Writer(const std::string& fname)
{
    const auto ios_mode = std::ios_base::out | std::ios_base::binary;

    if (ends_with(fname, ".maegz") || ends_with(fname, ".mae.gz")) {
        auto* gzip_stream = new filtering_ostream();
        gzip_stream->push(boost::iostreams::gzip_compressor());
        gzip_stream->push(file_sink(fname, ios_mode));
        m_out.reset(static_cast<ostream*>(gzip_stream));
    } else {
        auto* file_stream = new ofstream(fname, ios_mode);
        m_out.reset(static_cast<ostream*>(file_stream));
    }

    if(m_out->fail()) {
        std::stringstream ss;
        ss << "Failed to open file \"" << fname << "\" for writing operation.";
        throw std::runtime_error(ss.str());
    }

    write_opening_block();
}

void Writer::write(const std::shared_ptr<Block>& block)
{
    block->write(*m_out);
}

void Writer::write_opening_block()
{
    shared_ptr<Block> b = make_shared<Block>("");
    b->setStringProperty("s_m_m2io_version", "2.0.0");
    write(b);
}

} // namespace mae
} // namespace schrodinger
