#pragma once

#include "md_particle.hpp"

#include <zlib.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>


#pragma pack(push, 1)
struct FileHeader {
    char         magic[8];   // "PFRAME\0"
    std::uint32_t version;
    std::uint32_t reserved;
};

struct FrameHeader {
    std::uint64_t frame_index;
    std::uint64_t n_particles;
    std::uint64_t uncompressed_bytes;
    std::uint64_t compressed_bytes;
    std::uint32_t crc32;     // CRC of uncompressed Particle data
};
#pragma pack(pop)

// write/read exactly number of bytes to file
void write_exact(std::FILE* fp, const void* buf, std::size_t bytes);
bool read_exact(std::FILE* fp, void* buf, std::size_t bytes);


class FileWriter {
    public:
        FileWriter(const std::string& path, bool append);

        ~FileWriter();

        void write_frame(const Particle* data, std::size_t n_particles, std::uint64_t frame_index);
    
    private:
        std::FILE* fp_;
};

class FileReader {
    public:
        explicit FileReader(const std::string& path);

        ~FileReader();

        bool eof() const;

        bool corrupted() const;

        bool next_frame(std::vector<Particle>& particles, std::uint64_t& frame_index_out);
        
    private:
        std::FILE* fp_;

        bool eof_;

        bool corrupted_;

};