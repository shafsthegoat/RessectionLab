#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "simulation_control_native.hpp"

namespace resectionlab::sim {

// CPython 3.12.14 set probing/growth, specialized to admitted cell IDs (identity hashes).
// Algorithm reference, PSF-2.0: https://github.com/python/cpython/blob/v3.12.14/Objects/setobject.c
// shortcut: only IDs in [0, 2^24); extend admission and hash semantics before accepting other keys.
class PositiveIntSet {
public:
    explicit PositiveIntSet(const StopToken* stop = nullptr) : stop_(stop) {}

    template<class Iterator>
    PositiveIntSet(Iterator first, Iterator last, const StopToken* stop = nullptr) : stop_(stop) {
        update(first, last);
    }

    std::size_t size() const noexcept { return used_; }
    bool empty() const noexcept { return used_ == 0; }

    bool contains(std::int64_t value) const {
        if (value < 0 || value >= cell_limit) return false;
        return table_[position(value)] == value;
    }

    bool insert(std::int64_t value) {
        if (value < 0 || value >= cell_limit) throw std::out_of_range("Native integer set cell is out of range");
        const auto mask = table_.size() - 1;
        std::size_t i = static_cast<std::size_t>(value) & mask;
        std::size_t perturb = static_cast<std::size_t>(value), free = table_.size();
        for (;;) {
            const auto end = i + (i + linear_probes <= mask ? linear_probes : 0);
            for (auto slot = i; slot <= end; ++slot) {
                pulse();
                const auto key = table_[slot];
                if (key == value) return false;
                if (key == dummy) free = slot;
                if (key != unused) continue;
                if (free != table_.size()) {
                    table_[free] = value;
                    ++used_;
                } else {
                    table_[slot] = value;
                    ++used_; ++fill_;
                    if (fill_ * 5 >= mask * 3) resize(used_ * (used_ > 50000 ? 2 : 4));
                }
                return true;
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    bool erase(std::int64_t value) {
        if (value < 0 || value >= cell_limit) return false;
        const auto slot = position(value);
        if (table_[slot] != value) return false;
        table_[slot] = dummy;
        --used_;
        return true;
    }

    // Lists/tuples use incremental insertion; exact sets have a distinct pre-sizing path.
    template<class Iterator>
    void update(Iterator first, Iterator last) {
        for (; first != last; ++first) insert(*first);
        check();
    }

    void update_set(const PositiveIntSet& other) {
        if (this == &other || other.empty()) return;
        if ((fill_ + other.used_) * 5 >= (table_.size() - 1) * 3) resize((used_ + other.used_) * 2);
        if (fill_ == 0) {
            if (table_.size() == other.table_.size() && other.fill_ == other.used_) {
                for (std::size_t i = 0; i < table_.size(); ++i) { pulse(); table_[i] = other.table_[i]; }
            } else {
                other.for_each([&](std::int64_t value) { insert_clean(value); });
            }
            fill_ = used_ = other.used_;
        } else {
            other.for_each([&](std::int64_t value) { insert(value); });
        }
        check();
    }

    // Python set(s)/s.copy() may rehash; an ordinary C++ copy preserves the current table.
    PositiveIntSet copy() const {
        PositiveIntSet result(stop_);
        result.update_set(*this);
        return result;
    }

    PositiveIntSet united(const PositiveIntSet& other) const {
        auto result = copy();
        if (this != &other) result.update_set(other);
        return result;
    }

    PositiveIntSet difference(const PositiveIntSet& other) const {
        PositiveIntSet result(stop_);
        if ((used_ >> 2) > other.used_) {
            result = copy();
            other.for_each([&](std::int64_t value) { result.erase(value); });
            result.purge_dummies();
        } else {
            for_each([&](std::int64_t value) { if (!other.contains(value)) result.insert(value); });
        }
        result.check();
        return result;
    }

    template<class Iterator>
    void difference_update(Iterator first, Iterator last) {
        difference_update(first, last, [](std::int64_t value) { return value; });
    }

    template<class Iterator, class Key>
    void difference_update(Iterator first, Iterator last, Key key) {
        for (; first != last; ++first) erase(key(*first));
        purge_dummies();
        check();
    }

    template<class Function>
    void for_each(Function function) const {
        for (auto value : table_) { pulse(); if (value >= 0) function(value); }
        check();
    }

    std::vector<std::int64_t> ordered() const {
        std::vector<std::int64_t> result;
        result.reserve(used_);
        for_each([&](std::int64_t value) { result.push_back(value); });
        return result;
    }

private:
    static constexpr std::int64_t unused = -1, dummy = -2, cell_limit = 16777216;
    static constexpr std::size_t linear_probes = 9;
    std::vector<std::int64_t> table_ = std::vector<std::int64_t>(8, unused);
    std::size_t used_ = 0, fill_ = 0;
    const StopToken* stop_;
    mutable std::size_t primitives_ = 0;

    void check() const { if (stop_ != nullptr) stop_->check(); }
    void pulse() const { if ((++primitives_ & 32767) == 0) check(); }

    std::size_t position(std::int64_t value) const {
        const auto mask = table_.size() - 1;
        std::size_t i = static_cast<std::size_t>(value) & mask, perturb = static_cast<std::size_t>(value);
        for (;;) {
            const auto end = i + (i + linear_probes <= mask ? linear_probes : 0);
            for (auto slot = i; slot <= end; ++slot) {
                pulse();
                if (table_[slot] == unused || table_[slot] == value) return slot;
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    void insert_clean(std::int64_t value) {
        const auto mask = table_.size() - 1;
        std::size_t i = static_cast<std::size_t>(value) & mask, perturb = static_cast<std::size_t>(value);
        for (;;) {
            const auto end = i + (i + linear_probes <= mask ? linear_probes : 0);
            for (auto slot = i; slot <= end; ++slot) {
                pulse();
                if (table_[slot] == unused) { table_[slot] = value; return; }
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    void resize(std::size_t minimum) {
        std::size_t size = 8;
        while (size <= minimum) size *= 2;
        if (size == 8 && table_.size() == 8 && fill_ == used_) return;
        auto old = std::move(table_);
        table_.assign(size, unused);
        for (auto value : old) { pulse(); if (value >= 0) insert_clean(value); }
        fill_ = used_;
    }

    void purge_dummies() {
        if (fill_ - used_ > (table_.size() - 1) / 4) resize(used_ * (used_ > 50000 ? 2 : 4));
    }
};

}  // namespace resectionlab::sim
