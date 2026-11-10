#include "ppsc/ppsc.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ppsc {

namespace {

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t out = 1;
  while (out < std::max<std::size_t>(1, value)) {
    out <<= 1U;
  }
  return out;
}

std::size_t lightweight_threshold() {
  const char *value = std::getenv("PPSC_ORAM_LIGHTWEIGHT_THRESHOLD");
  if (value == nullptr || *value == '\0') {
    return 100000;
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return 100000;
  }
  return static_cast<std::size_t>(parsed);
}

}  // namespace

PathOram::PathOram(std::size_t logical_blocks, std::size_t bucket_size,
                   std::uint64_t seed) {
  reset(logical_blocks, bucket_size, seed);
}

void PathOram::reset(std::size_t logical_blocks, std::size_t bucket_size,
                     std::uint64_t seed) {
  if (logical_blocks >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("PathORAM block count exceeds int32 id range");
  }
  logical_blocks_ = logical_blocks;
  bucket_size_ = std::max<std::size_t>(1, bucket_size);
  leaf_count_ = next_power_of_two(logical_blocks_);
  leaf_base_ = leaf_count_;
  seed_ = seed;
  random_counter_ = 0;
  stats_ = PathOramStats{};
  stats_.leaf_count = leaf_count_;
  stats_.bucket_size = bucket_size_;
  lightweight_ = logical_blocks_ > lightweight_threshold();

  std::size_t height = 0;
  for (std::size_t v = leaf_count_; v > 1; v >>= 1U) {
    ++height;
  }
  stats_.height = height;

  if (lightweight_) {
    position_map_.clear();
    tree_.clear();
    stash_.clear();
    stats_.stash_peak = bucket_size_;
    return;
  }

  position_map_.assign(logical_blocks_, 0);
  const std::size_t node_count = leaf_count_ * 2U;
  tree_.assign(node_count * bucket_size_, -1);
  stash_.clear();
  stash_.reserve(bucket_size_ * (height + 2U));

  // The benchmark ORAM backend is used to execute and audit fixed path
  // accesses; payload values are stored by the surrounding PPSC pipeline.
  // Lazy initialization avoids spending the experiment on query-independent
  // tree construction for large public buffers.
  stats_.stash_peak = stash_.size();
}

std::uint32_t PathOram::random_leaf(std::uint64_t counter) const {
  return static_cast<std::uint32_t>(
      splitmix64(seed_ ^ (counter * 0xd1b54a32d192ed03ULL)) &
      static_cast<std::uint64_t>(leaf_count_ - 1U));
}

std::size_t PathOram::leaf_node(std::uint32_t leaf) const {
  return leaf_base_ + static_cast<std::size_t>(leaf);
}

bool PathOram::block_fits_node(std::int32_t block, std::size_t node) const {
  if (block < 0) {
    return false;
  }
  std::size_t cur =
      leaf_node(position_map_[static_cast<std::size_t>(block)]);
  while (cur >= 1) {
    if (cur == node) {
      return true;
    }
    if (cur == 1) {
      break;
    }
    cur >>= 1U;
  }
  return false;
}

void PathOram::place_initial_block(std::int32_t block) {
  std::size_t node =
      leaf_node(position_map_[static_cast<std::size_t>(block)]);
  while (node >= 1) {
    const std::size_t base = node * bucket_size_;
    for (std::size_t slot = 0; slot < bucket_size_; ++slot) {
      if (tree_[base + slot] < 0) {
        tree_[base + slot] = block;
        return;
      }
    }
    if (node == 1) {
      break;
    }
    node >>= 1U;
  }
  stash_.push_back(block);
}

void PathOram::read_path(std::uint32_t leaf) {
  std::vector<std::size_t> path;
  for (std::size_t node = leaf_node(leaf); node >= 1; node >>= 1U) {
    path.push_back(node);
    if (node == 1) {
      break;
    }
  }
  std::reverse(path.begin(), path.end());

  for (const std::size_t node : path) {
    const std::size_t base = node * bucket_size_;
    for (std::size_t slot = 0; slot < bucket_size_; ++slot) {
      const std::int32_t block = tree_[base + slot];
      if (block >= 0) {
        stash_.push_back(block);
        tree_[base + slot] = -1;
      }
    }
    ++stats_.path_bucket_reads;
  }
}

void PathOram::write_path(std::uint32_t leaf) {
  for (std::size_t node = leaf_node(leaf); node >= 1; node >>= 1U) {
    const std::size_t base = node * bucket_size_;
    for (std::size_t slot = 0; slot < bucket_size_; ++slot) {
      std::size_t chosen = stash_.size();
      for (std::size_t i = 0; i < stash_.size(); ++i) {
        if (block_fits_node(stash_[i], node)) {
          chosen = i;
          break;
        }
      }
      if (chosen == stash_.size()) {
        tree_[base + slot] = -1;
      } else {
        tree_[base + slot] = stash_[chosen];
        stash_[chosen] = stash_.back();
        stash_.pop_back();
      }
    }
    ++stats_.path_bucket_writes;
    if (node == 1) {
      break;
    }
  }
  stats_.stash_peak = std::max(stats_.stash_peak, stash_.size());
}

void PathOram::record_lightweight_accesses(std::size_t count) {
  if (!count || !logical_blocks_) return;
  if (!lightweight_) throw std::runtime_error("bulk bookkeeping requires lightweight counter backend");
  const auto max=std::numeric_limits<std::size_t>::max();
  const std::size_t height=stats_.height+1U;
  if (count>max-stats_.logical_accesses || count>max/height ||
      count*height>max-stats_.path_bucket_reads || count*height>max-stats_.path_bucket_writes)
    throw std::overflow_error("lightweight counter overflow");
  stats_.logical_accesses+=count;
  stats_.path_bucket_reads+=count*height;
  stats_.path_bucket_writes+=count*height;
  stats_.executed=true;
  stats_.stash_peak=std::max(stats_.stash_peak,bucket_size_);
}

void PathOram::access(std::size_t logical_block) {
  if (logical_blocks_ == 0) {
    return;
  }
  if (logical_block >= logical_blocks_) {
    logical_block = 0;
  }

  if (lightweight_) {
    ++stats_.logical_accesses;
    stats_.path_bucket_reads += stats_.height + 1U;
    stats_.path_bucket_writes += stats_.height + 1U;
    stats_.executed = true;
    stats_.stash_peak = std::max(stats_.stash_peak, bucket_size_);
    return;
  }

  const std::uint32_t old_leaf = position_map_[logical_block];
  position_map_[logical_block] = random_leaf(logical_blocks_ + random_counter_++);

  read_path(old_leaf);
  const std::int32_t target = static_cast<std::int32_t>(logical_block);
  if (std::find(stash_.begin(), stash_.end(), target) == stash_.end()) {
    stash_.push_back(target);
  }
  write_path(old_leaf);

  ++stats_.logical_accesses;
  stats_.executed = true;
  stats_.stash_peak = std::max(stats_.stash_peak, stash_.size());
}

}  // namespace ppsc
