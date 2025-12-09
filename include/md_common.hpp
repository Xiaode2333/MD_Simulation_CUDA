#pragma once

#include "md_particle.hpp"

#include <zlib.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <fmt/core.h>
#include <utility>
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

template <typename... Args>
void RankZeroPrint(int rank_idx, fmt::format_string<Args...> format_str, Args&&... args) {
    if (rank_idx == 0) {
        fmt::print(format_str, std::forward<Args>(args)...);
        std::fflush(stdout);
    }
}

// Create directory tree if it does not exist (rank 0 only). Returns true on success.
bool create_folder(const std::filesystem::path& path, int rank_idx);

// Append the last non-empty line from src to dst (rank 0 only). `tag` is used as log prefix.
bool append_latest_line(const std::filesystem::path& src,
                        const std::filesystem::path& dst,
                        int rank_idx,
                        const std::string& tag);
                        
// Write header and density values to CSV (rank 0 only). Overwrites existing file.
void write_density_profile_csv(const std::filesystem::path& filepath,
                               const std::vector<double>& density,
                               int rank_idx,
                               const std::string& tag);

// Append formatted CSV text (rank 0 only). Caller supplies trailing newline in format string.
// Examples:
//   append_csv("foo.csv", rank_idx, "tag", "{},{}\n", id, value);
//   append_csv("row.csv", rank_idx, "tag", "row_id,{}\n", fmt::join(values, ",")); // values {1,2,3} -> "1,2,3"
//   append_csv("row.csv", rank_idx, "tag", "{}\n", fmt::join(str_values, ","));    // str_values {"a","b"} -> "a,b"
template <typename... Args>
bool append_csv(const std::filesystem::path& filepath,
                int rank_idx,
                const std::string& tag,
                fmt::format_string<Args...> fmt_str,
                Args&&... args) {
    if (rank_idx != 0) {
        return true;
    }

    std::ofstream out(filepath, std::ios::out | std::ios::app);
    if (!out) {
        fmt::print(stderr, "[{}] Failed to open {} for appending.\n", tag, filepath.string());
        return false;
    }

    out << fmt::format(fmt_str, std::forward<Args>(args)...);
    out.flush();
    return true;
}

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
