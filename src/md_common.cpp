#include "md_common.hpp"
#include "md_particle.hpp"

#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <string>

bool create_folder(const std::filesystem::path &path, int rank_idx) {
  if (rank_idx != 0) {
    return true;
  }

  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    std::filesystem::create_directories(path, ec);
    if (ec) {
      fmt::print(stderr, "Failed to create dir {}. Error: {}\n", path.string(),
                 ec.message());
      return false;
    }
  }
  return true;
}

bool append_latest_line(const std::filesystem::path &src,
                        const std::filesystem::path &dst, int rank_idx,
                        const std::string &tag) {
  if (rank_idx != 0) {
    return true;
  }

  std::ifstream in(src);
  if (!in) {
    fmt::print(stderr, "[{}] Failed to open {} for reading.\n", tag,
               src.string());
    return false;
  }

  std::string line;
  std::string last_non_empty;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      last_non_empty = line;
    }
  }

  if (last_non_empty.empty()) {
    fmt::print(stderr, "[{}] No data found in {}.\n", tag, src.string());
    return false;
  }

  std::ofstream out(dst, std::ios::out | std::ios::app);
  if (!out) {
    fmt::print(stderr, "[{}] Failed to open {} for appending.\n", tag,
               dst.string());
    return false;
  }

  out << last_non_empty << '\n';
  return true;
}

void write_density_profile_csv(const std::filesystem::path &filepath,
                               const std::vector<double> &density, int rank_idx,
                               const std::string &tag) {
  if (rank_idx != 0) {
    return;
  }

  std::ofstream out(filepath, std::ios::out | std::ios::trunc);
  if (!out) {
    fmt::print(stderr, "[{}] Failed to open {} for writing density profile.\n",
               tag, filepath.string());
    return;
  }

  out << "bin,rho\n";
  for (std::size_t idx = 0; idx < density.size(); ++idx) {
    out << idx << ',' << density[idx] << '\n';
  }
}

void write_exact(std::FILE *fp, const void *buf, std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  std::size_t written = std::fwrite(buf, 1, bytes, fp);
  if (written != bytes) {
    throw std::runtime_error("Failed to write to file");
  }
}

bool read_exact(std::FILE *fp, void *buf, std::size_t bytes) {
  if (bytes == 0) {
    return true;
  }
  std::size_t read = std::fread(buf, 1, bytes, fp);
  if (read == 0) {
    // Pure EOF, not an error
    return false;
  }
  if (read != bytes) {
    // Partial read: broken tail
    return false;
  }
  return true;
}

FileWriter::FileWriter(const std::string &path, bool append) {
  fp_ = nullptr;
  if (append) {
    // Open existing file for update, verify header, then seek to end.
    fp_ = std::fopen(path.c_str(), "r+b");
    if (!fp_) {
      throw std::runtime_error("Failed to open file for append: " + path);
    }

    FileHeader header{};
    if (!read_exact(fp_, &header, sizeof(header))) {
      std::fclose(fp_);
      throw std::runtime_error("Failed to read existing header for append");
    }

    if (std::memcmp(header.magic, "PFRAME", 6) != 0 ||
        (header.version != 1 && header.version != 2)) {
      std::fclose(fp_);
      throw std::runtime_error("Not a valid PFRAME v1/v2 file");
    }

    // Seek to end for appending new frames
    if (std::fseek(fp_, 0, SEEK_END) != 0) {
      std::fclose(fp_);
      throw std::runtime_error("Failed to seek to end of file");
    }
  } else {
    fp_ = std::fopen(path.c_str(), "wb");
    if (!fp_) {
      throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write new file header
    FileHeader header{};
    std::memset(header.magic, 0, sizeof(header.magic));
    std::memcpy(header.magic, "PFRAME", 6);
    header.version = 2; // Version 2: Added theta field to Particle struct
    header.reserved = 0;

    write_exact(fp_, &header, sizeof(header));
    std::fflush(fp_);
  }
}

FileWriter::~FileWriter() {
  if (fp_) {
    std::fflush(fp_);
    std::fclose(fp_);
  }
}

void FileWriter::write_frame(const Particle *data, std::size_t n_particles,
                             std::uint64_t frame_index) {
  if (!fp_) {
    throw std::runtime_error("File not open");
  }

  std::uint64_t uncompressed_bytes =
      static_cast<std::uint64_t>(n_particles) * sizeof(Particle);

  // Build uncompressed byte view
  const unsigned char *raw = reinterpret_cast<const unsigned char *>(data);

  // Compute CRC32 of uncompressed data
  std::uint32_t crc = ::crc32(0L, Z_NULL, 0);
  crc = ::crc32(crc, raw, static_cast<uInt>(uncompressed_bytes));

  // Allocate buffer for compressed data
  uLong source_len = static_cast<uLong>(uncompressed_bytes);
  uLong dest_bound = ::compressBound(source_len);
  std::vector<unsigned char> compressed(dest_bound);

  uLongf dest_len = dest_bound;
  int zret = ::compress2(compressed.data(), &dest_len, raw, source_len,
                         Z_BEST_COMPRESSION);
  if (zret != Z_OK) {
    throw std::runtime_error("compress2 failed");
  }

  std::uint64_t compressed_bytes = static_cast<std::uint64_t>(dest_len);

  FrameHeader fh{};
  fh.frame_index = frame_index;
  fh.n_particles = static_cast<std::uint64_t>(n_particles);
  fh.uncompressed_bytes = uncompressed_bytes;
  fh.compressed_bytes = compressed_bytes;
  fh.crc32 = crc;

  // Write frame header then compressed payload
  write_exact(fp_, &fh, sizeof(fh));
  write_exact(fp_, compressed.data(),
              static_cast<std::size_t>(compressed_bytes));

  std::fflush(fp_);
}

FileReader::FileReader(const std::string &path) {
  fp_ = nullptr;
  eof_ = false;
  corrupted_ = false;
  version_ = 0;
  fp_ = std::fopen(path.c_str(), "rb");
  if (!fp_) {
    throw std::runtime_error("Failed to open file for reading: " + path);
  }

  FileHeader header{};
  if (!read_exact(fp_, &header, sizeof(header))) {
    std::fclose(fp_);
    throw std::runtime_error("Failed to read file header");
  }

  if (std::memcmp(header.magic, "PFRAME", 6) != 0) {
    std::fclose(fp_);
    throw std::runtime_error("Not a valid PFRAME file");
  }

  // Store version
  version_ = header.version;

  // Reject v1 files - incompatible with new Particle struct (added theta field)
  if (version_ == 1) {
    std::fclose(fp_);
    throw std::runtime_error(
        "File format v1 is incompatible with current build (v2).\n"
        "  Reason: Particle struct now includes 'theta' field for ABP.\n"
        "  Solution: Regenerate simulation data with v2 format.");
  }

  if (version_ != 2) {
    std::fclose(fp_);
    throw std::runtime_error("Unknown file version: " +
                             std::to_string(version_));
  }
}

FileReader::~FileReader() {
  if (fp_) {
    std::fclose(fp_);
  }
}

bool FileReader::eof() const { return eof_; }

bool FileReader::corrupted() const { return corrupted_; }

bool FileReader::next_frame(std::vector<Particle> &particles,
                            std::uint64_t &frame_index_out) {
  if (!fp_ || eof_) {
    return false;
  }

  FrameHeader fh{};
  if (!read_exact(fp_, &fh, sizeof(fh))) {
    // Either clean EOF (no bytes) or broken partial header.
    eof_ = true;
    return false;
  }

  // Basic sanity checks
  if (fh.uncompressed_bytes != fh.n_particles * sizeof(Particle) ||
      fh.compressed_bytes == 0) {
    corrupted_ = true;
    eof_ = true;
    return false;
  }

  // Read compressed payload
  std::vector<unsigned char> compressed(
      static_cast<std::size_t>(fh.compressed_bytes));

  if (!read_exact(fp_, compressed.data(),
                  static_cast<std::size_t>(fh.compressed_bytes))) {
    // Truncated compressed data: broken tail
    corrupted_ = true;
    eof_ = true;
    return false;
  }

  // Decompress
  std::vector<unsigned char> uncompressed(
      static_cast<std::size_t>(fh.uncompressed_bytes));

  uLongf dest_len = static_cast<uLongf>(fh.uncompressed_bytes);
  int zret = ::uncompress(uncompressed.data(), &dest_len, compressed.data(),
                          static_cast<uLong>(fh.compressed_bytes));
  if (zret != Z_OK || dest_len != fh.uncompressed_bytes) {
    corrupted_ = true;
    eof_ = true;
    return false;
  }

  // Check CRC
  std::uint32_t crc = ::crc32(0L, Z_NULL, 0);
  crc = ::crc32(crc, uncompressed.data(),
                static_cast<uInt>(fh.uncompressed_bytes));
  if (crc != fh.crc32) {
    corrupted_ = true;
    eof_ = true;
    return false;
  }

  // Interpret as Particle[]
  std::size_t n_particles = static_cast<std::size_t>(fh.n_particles);
  particles.resize(n_particles);
  std::memcpy(particles.data(), uncompressed.data(),
              static_cast<std::size_t>(fh.uncompressed_bytes));

  frame_index_out = fh.frame_index;
  return true;
};
