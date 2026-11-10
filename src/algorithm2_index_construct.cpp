#include "ppsc/ppsc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ppsc {

namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const std::string &label) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::runtime_error("size overflow while computing " + label);
  }
  return a * b;
}

void read_exact(std::ifstream &input, void *buffer, std::uint64_t bytes,
                const std::string &path) {
  input.read(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("short read from SuCo index: " + path);
  }
}

void seek_to(std::ifstream &input, std::uint64_t offset,
             const std::string &path) {
  input.clear();
  input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!input) {
    throw std::runtime_error("cannot seek SuCo index: " + path);
  }
}

std::uint64_t file_size(std::ifstream &input, const std::string &path) {
  input.clear();
  input.seekg(0, std::ios::end);
  if (!input) {
    throw std::runtime_error("cannot inspect SuCo index: " + path);
  }
  const auto size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("invalid SuCo index size: " + path);
  }
  return static_cast<std::uint64_t>(size);
}

std::size_t cell_id(std::int32_t first, std::int32_t second,
                    std::size_t centroid_count) {
  if (first < 0 || second < 0 ||
      static_cast<std::size_t>(first) >= centroid_count ||
      static_cast<std::size_t>(second) >= centroid_count) {
    throw std::runtime_error("assignment id exceeds the configured centroid count");
  }
  return static_cast<std::size_t>(first) * centroid_count +
         static_cast<std::size_t>(second);
}

}  // namespace

PublicPaddedIndex algorithm2_build_public_padded_index(
    const BenchmarkConfig &config) {
  PublicPaddedIndex index;
  index.dataset_size = config.dataset_size;
  index.bucket_size = std::max<std::size_t>(1, config.bucket_size);
  index.bucket_count =
      (config.dataset_size + index.bucket_size - 1) / index.bucket_size;
  index.padded_slots = index.bucket_count * index.bucket_size;
  index.upstream_index_ms = config.upstream_index_ms;

  if (config.suco_index_path.empty()) {
    std::cerr << "No SuCo index path provided; using metadata-only padded index.\n";
    return index;
  }

  if (config.subspace_num == 0 || config.kmeans_num_centroid == 0) {
    throw std::runtime_error("subspace_num and kmeans_num_centroid must be positive");
  }

  index.subspace_num = config.subspace_num;
  index.subspace_dimensionality =
      config.subspace_dimensionality == 0
          ? config.dim / config.subspace_num
          : config.subspace_dimensionality;
  if (index.subspace_dimensionality == 0 ||
      index.subspace_dimensionality % 2 != 0) {
    throw std::runtime_error("subspace_dimensionality must be positive and even");
  }
  if (index.subspace_num * index.subspace_dimensionality > config.dim) {
    throw std::runtime_error("subspace schedule exceeds input dimensionality");
  }
  index.kmeans_dim = index.subspace_dimensionality / 2;
  index.kmeans_num_centroid = config.kmeans_num_centroid;
  index.cell_count = checked_product(index.kmeans_num_centroid,
                                     index.kmeans_num_centroid, "cell_count");
  index.public_bucket_count =
      checked_product(index.subspace_num, index.cell_count, "public_bucket_count");
  index.real_index_entries =
      checked_product(index.subspace_num, index.dataset_size, "real_index_entries");

  const std::uint64_t centroid_count =
      static_cast<std::uint64_t>(index.subspace_num) * 2ULL *
      static_cast<std::uint64_t>(index.kmeans_num_centroid) *
      static_cast<std::uint64_t>(index.kmeans_dim);
  const std::uint64_t centroid_bytes = centroid_count * sizeof(float);
  const std::uint64_t assignment_count =
      static_cast<std::uint64_t>(index.subspace_num) * 2ULL *
      static_cast<std::uint64_t>(index.dataset_size);
  const std::uint64_t assignment_bytes = assignment_count * sizeof(std::int32_t);

  std::ifstream input(config.suco_index_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open SuCo index: " + config.suco_index_path);
  }
  const std::uint64_t actual_size = file_size(input, config.suco_index_path);
  if (actual_size < centroid_bytes + assignment_bytes) {
    throw std::runtime_error("SuCo index is smaller than the configured layout: " +
                             config.suco_index_path);
  }

  const auto start = std::chrono::steady_clock::now();

  index.centroids.resize(static_cast<std::size_t>(centroid_count));
  seek_to(input, 0, config.suco_index_path);
  read_exact(input, index.centroids.data(), centroid_bytes, config.suco_index_path);

  index.bucket_counts.assign(index.public_bucket_count, 0);
  std::vector<std::int32_t> first(index.dataset_size);
  std::vector<std::int32_t> second(index.dataset_size);

  std::cerr << "Materializing PPSC padded public index from SuCo index: "
            << config.suco_index_path << '\n';

  for (std::size_t subspace = 0; subspace < index.subspace_num; ++subspace) {
    const std::uint64_t base =
        centroid_bytes +
        static_cast<std::uint64_t>(subspace) * 2ULL *
            static_cast<std::uint64_t>(index.dataset_size) * sizeof(std::int32_t);
    seek_to(input, base, config.suco_index_path);
    read_exact(input, first.data(),
               static_cast<std::uint64_t>(index.dataset_size) * sizeof(std::int32_t),
               config.suco_index_path);
    read_exact(input, second.data(),
               static_cast<std::uint64_t>(index.dataset_size) * sizeof(std::int32_t),
               config.suco_index_path);

    const std::size_t bucket_base = subspace * index.cell_count;
    for (std::size_t id = 0; id < index.dataset_size; ++id) {
      const std::size_t c =
          cell_id(first[id], second[id], index.kmeans_num_centroid);
      ++index.bucket_counts[bucket_base + c];
    }
  }

  index.subspace_min_cell_sizes.assign(index.subspace_num, 0);
  index.subspace_max_cell_sizes.assign(index.subspace_num, 0);
  std::uint32_t global_max_capacity = 0;
  for (std::size_t subspace = 0; subspace < index.subspace_num; ++subspace) {
    const std::size_t begin = subspace * index.cell_count;
    const std::size_t end = begin + index.cell_count;
    std::uint32_t observed_min = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t observed_max = 0;
    for (std::size_t bucket = begin; bucket < end; ++bucket) {
      const std::uint32_t count = index.bucket_counts[bucket];
      if (count > 0) {
        observed_min = std::min(observed_min, count);
      }
      observed_max = std::max(observed_max, count);
    }
    if (observed_min == std::numeric_limits<std::uint32_t>::max()) {
      observed_min = 1;
    }
    const std::uint32_t capacity =
        config.padded_bucket_capacity == 0
            ? std::max<std::uint32_t>(1, observed_max)
            : static_cast<std::uint32_t>(config.padded_bucket_capacity);
    if (capacity < observed_max)
      throw std::runtime_error("padded-bucket-capacity would truncate real identifiers");
    index.subspace_min_cell_sizes[subspace] = observed_min;
    index.subspace_max_cell_sizes[subspace] = capacity;
    global_max_capacity = std::max(global_max_capacity, capacity);
  }
  index.padded_bucket_capacity = global_max_capacity == 0 ? 1 : global_max_capacity;

  index.bucket_offsets.resize(index.public_bucket_count + 1);
  std::uint64_t next_offset = 0;
  for (std::size_t bucket = 0; bucket < index.public_bucket_count; ++bucket) {
    const std::size_t subspace = bucket / index.cell_count;
    index.bucket_offsets[bucket] = next_offset;
    next_offset += static_cast<std::uint64_t>(
        index.subspace_max_cell_sizes[subspace]);
  }
  index.bucket_offsets[index.public_bucket_count] = next_offset;
  if (index.bucket_offsets.back() >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("padded index is too large for this platform");
  }

  index.padded_index_slots = static_cast<std::size_t>(index.bucket_offsets.back());
  index.dummy_index_entries =
      index.padded_index_slots > index.real_index_entries
          ? index.padded_index_slots - index.real_index_entries
          : 0;
  index.padded_bucket_ids.assign(index.padded_index_slots, -1);

  std::vector<std::uint32_t> cursors(index.public_bucket_count, 0);
  seek_to(input, centroid_bytes, config.suco_index_path);
  for (std::size_t subspace = 0; subspace < index.subspace_num; ++subspace) {
    read_exact(input, first.data(),
               static_cast<std::uint64_t>(index.dataset_size) * sizeof(std::int32_t),
               config.suco_index_path);
    read_exact(input, second.data(),
               static_cast<std::uint64_t>(index.dataset_size) * sizeof(std::int32_t),
               config.suco_index_path);

    const std::size_t bucket_base = subspace * index.cell_count;
    for (std::size_t id = 0; id < index.dataset_size; ++id) {
      const std::size_t bucket =
          bucket_base + cell_id(first[id], second[id], index.kmeans_num_centroid);
      const std::uint32_t pos = cursors[bucket]++;
      const std::uint64_t capacity =
          index.bucket_offsets[bucket + 1] - index.bucket_offsets[bucket];
      if (pos >= capacity) {
        ++index.truncated_index_entries;
        continue;
      }
      const std::uint64_t out =
          index.bucket_offsets[bucket] + static_cast<std::uint64_t>(pos);
      index.padded_bucket_ids[static_cast<std::size_t>(out)] =
          static_cast<std::int32_t>(id);
    }
  }

  const auto end = std::chrono::steady_clock::now();
  index.padding_materialization_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cerr << "PPSC public buckets: " << index.public_bucket_count
            << ", max padded bucket capacity: " << index.padded_bucket_capacity
            << ", padded index slots: " << index.padded_index_slots
            << ", dummy slots: " << index.dummy_index_entries
            << ", truncated entries: " << index.truncated_index_entries << '\n';
  return index;
}

}  // namespace ppsc
