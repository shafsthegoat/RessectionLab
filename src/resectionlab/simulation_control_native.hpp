#pragma once

#include <atomic>
#include <chrono>
#include <stdexcept>

namespace resectionlab::sim {

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("SIMULATION_CANCELLED") {}
};
struct DeadlineExceeded : std::runtime_error {
    DeadlineExceeded() : std::runtime_error("SIMULATION_DEADLINE_EXCEEDED") {}
};

class StopToken {
public:
    using Clock = std::chrono::steady_clock;

    void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
    void deadline(Clock::time_point value) noexcept { deadline_.store(value.time_since_epoch().count(),std::memory_order_relaxed); }
    void check() const {
        if (cancelled_.load(std::memory_order_relaxed)) throw Cancelled();
        if (Clock::now().time_since_epoch().count() >= deadline_.load(std::memory_order_relaxed))
            throw DeadlineExceeded();
    }

private:
    std::atomic<bool> cancelled_{false};
    std::atomic<Clock::duration::rep> deadline_{Clock::duration::max().count()};
};

}  // namespace resectionlab::sim
