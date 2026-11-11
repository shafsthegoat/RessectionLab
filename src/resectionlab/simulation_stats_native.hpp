#pragma once
#include <atomic>
#include <cstdint>

namespace resectionlab::sim {
struct ExecutionStats {
    std::atomic<std::uint64_t> actions{0}, trajectories{0}, physical_substeps{0}, clones{0}, rollbacks{0}, state_hashes{0};
    std::atomic<std::uint64_t> geometry_batches{0}, geometry_batch_items{0}, geometry_batch_max_items{0};
};
inline ExecutionStats execution_stats;
}  // namespace resectionlab::sim
