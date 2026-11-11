#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "simulation_value_native.hpp"

namespace resectionlab::sim {

// Exact binary capacity, stored as a signed integer coefficient times 2^exponent.
class Dyadic {
    using Limbs = std::vector<std::uint32_t>;
public:
    Dyadic() = default;

    static Dyadic from_double(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite native dyadic value");
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const unsigned encoded_exponent = static_cast<unsigned>((bits >> 52) & 0x7ff);
        std::uint64_t coefficient = bits & ((std::uint64_t(1) << 52) - 1);
        int exponent = -1074;
        if (encoded_exponent != 0) {
            coefficient |= std::uint64_t(1) << 52;
            exponent = static_cast<int>(encoded_exponent) - 1023 - 52;
        }
        return Dyadic((bits >> 63) ? -1 : 1, words(coefficient), exponent);
    }

    static Dyadic from_pair(const Value& pair) {
        if (!pair.is_array() || pair.as_array().size() != 2)
            throw std::invalid_argument("NATIVE_CUSA_NON_DYADIC_PROGRESS");
        auto numerator = integer(pair.at(0));
        auto denominator = integer(pair.at(1));
        const int power = power_of_two(denominator.second);
        return Dyadic(numerator.first * denominator.first, std::move(numerator.second), -power);
    }

    Value pair() const {
        auto numerator = coefficient_;
        Limbs denominator = words(1);
        if (exponent_ >= 0) numerator = shifted(numerator, static_cast<std::size_t>(exponent_));
        else denominator = shifted(denominator, static_cast<std::size_t>(-std::int64_t(exponent_)));
        return Value::array({integer_value(sign_, numerator), integer_value(1, denominator)});
    }

    bool zero() const noexcept { return sign_ == 0; }
    bool positive() const noexcept { return sign_ > 0; }

    double to_double() const {
        if (zero()) return 0.;
        const auto count = bit_count(coefficient_);
        std::int64_t highest = std::int64_t(exponent_) + static_cast<std::int64_t>(count) - 1;
        std::uint64_t encoded;
        if (highest >= -1022) {
            if (highest > 1023) throw std::overflow_error("integer division result too large for a float");
            auto mantissa = rounded_shift(static_cast<std::int64_t>(count) - 53);
            if (mantissa == (std::uint64_t(1) << 53)) { mantissa >>= 1; ++highest; }
            if (highest > 1023) throw std::overflow_error("integer division result too large for a float");
            encoded = (static_cast<std::uint64_t>(highest + 1023) << 52) |
                (mantissa & ((std::uint64_t(1) << 52) - 1));
        } else {
            encoded = rounded_shift(-std::int64_t(exponent_) - 1074);
        }
        if (sign_ < 0) encoded |= std::uint64_t(1) << 63;
        double result;
        std::memcpy(&result, &encoded, sizeof(result));
        return result;
    }

    friend Dyadic operator+(const Dyadic& first, const Dyadic& second) {
        if (first.zero()) return second;
        if (second.zero()) return first;
        const int exponent = std::min(first.exponent_, second.exponent_);
        auto left = shifted(first.coefficient_, std::size_t(std::int64_t(first.exponent_) - exponent));
        auto right = shifted(second.coefficient_, std::size_t(std::int64_t(second.exponent_) - exponent));
        if (first.sign_ == second.sign_) return Dyadic(first.sign_, add(left, right), exponent);
        const int order = magnitude_compare(left, right);
        if (order == 0) return {};
        return order > 0 ? Dyadic(first.sign_, subtract(left, right), exponent) :
            Dyadic(second.sign_, subtract(right, left), exponent);
    }

    friend Dyadic operator-(const Dyadic& first, Dyadic second) {
        second.sign_ = -second.sign_;
        return first + second;
    }

    friend int compare(const Dyadic& first, const Dyadic& second) {
        if (first.sign_ != second.sign_) return first.sign_ < second.sign_ ? -1 : 1;
        if (first.zero()) return 0;
        const auto left_bits = std::int64_t(bit_count(first.coefficient_)) + first.exponent_;
        const auto right_bits = std::int64_t(bit_count(second.coefficient_)) + second.exponent_;
        if (left_bits != right_bits) return (left_bits < right_bits ? -1 : 1) * first.sign_;
        const int exponent = std::min(first.exponent_, second.exponent_);
        return magnitude_compare(shifted(first.coefficient_, std::size_t(std::int64_t(first.exponent_) - exponent)),
            shifted(second.coefficient_, std::size_t(std::int64_t(second.exponent_) - exponent))) * first.sign_;
    }

    friend bool operator<(const Dyadic& first, const Dyadic& second) { return compare(first, second) < 0; }
    friend bool operator>=(const Dyadic& first, const Dyadic& second) { return compare(first, second) >= 0; }

private:
    Dyadic(int sign, Limbs coefficient, int exponent)
        : sign_(sign), coefficient_(std::move(coefficient)), exponent_(exponent) { normalize(); }

    static Limbs words(std::uint64_t value) {
        if (value == 0) return {};
        Limbs result{static_cast<std::uint32_t>(value)};
        if (value >> 32) result.push_back(static_cast<std::uint32_t>(value >> 32));
        return result;
    }
    static void trim(Limbs& value) { while (!value.empty() && value.back() == 0) value.pop_back(); }
    static unsigned trailing_zeros(std::uint32_t value) {
        unsigned result = 0;
        while ((value & 1) == 0) { ++result; value >>= 1; }
        return result;
    }
    static std::size_t bit_count(const Limbs& value) {
        if (value.empty()) return 0;
        unsigned count = 0;
        for (auto highest = value.back(); highest != 0; highest >>= 1) ++count;
        return (value.size() - 1) * 32 + count;
    }
    void normalize() {
        trim(coefficient_);
        if (coefficient_.empty()) { sign_ = 0; exponent_ = 0; return; }
        std::size_t whole = 0;
        while (coefficient_[whole] == 0) ++whole;
        const unsigned partial = trailing_zeros(coefficient_[whole]);
        const std::int64_t next_exponent = std::int64_t(exponent_) +
            static_cast<std::int64_t>(whole) * 32 + partial;
        if (next_exponent > std::numeric_limits<int>::max())
            throw std::overflow_error("Native dyadic exponent overflow");
        exponent_ = static_cast<int>(next_exponent);
        coefficient_.erase(coefficient_.begin(), coefficient_.begin() + whole);
        if (partial != 0) {
            for (std::size_t index = 0; index < coefficient_.size(); ++index) {
                coefficient_[index] = (coefficient_[index] >> partial) |
                    (index + 1 < coefficient_.size() ? coefficient_[index + 1] << (32 - partial) : 0);
            }
            trim(coefficient_);
        }
    }
    static Limbs shifted(const Limbs& value, std::size_t bits) {
        if (value.empty()) return {};
        const auto whole = bits / 32;
        const unsigned partial = static_cast<unsigned>(bits % 32);
        Limbs result(whole + value.size() + (partial != 0), 0);
        std::uint32_t carry = 0;
        for (std::size_t index = 0; index < value.size(); ++index) {
            result[whole + index] = (value[index] << partial) | carry;
            carry = partial ? value[index] >> (32 - partial) : 0;
        }
        if (partial) result.back() = carry;
        trim(result);
        return result;
    }
    static int magnitude_compare(const Limbs& first, const Limbs& second) {
        if (first.size() != second.size()) return first.size() < second.size() ? -1 : 1;
        for (std::size_t index = first.size(); index != 0;) {
            --index;
            if (first[index] != second[index]) return first[index] < second[index] ? -1 : 1;
        }
        return 0;
    }
    static Limbs add(const Limbs& first, const Limbs& second) {
        Limbs result(std::max(first.size(), second.size()));
        std::uint64_t carry = 0;
        for (std::size_t index = 0; index < result.size(); ++index) {
            const std::uint64_t sum = carry + (index < first.size() ? first[index] : 0) +
                (index < second.size() ? std::uint64_t(second[index]) : 0);
            result[index] = static_cast<std::uint32_t>(sum); carry = sum >> 32;
        }
        if (carry) result.push_back(static_cast<std::uint32_t>(carry));
        return result;
    }
    static Limbs subtract(const Limbs& first, const Limbs& second) {
        Limbs result(first.size());
        std::uint64_t borrow = 0;
        for (std::size_t index = 0; index < first.size(); ++index) {
            const std::uint64_t right = borrow + (index < second.size() ? std::uint64_t(second[index]) : 0);
            result[index] = static_cast<std::uint32_t>(std::uint64_t(first[index]) - right);
            borrow = std::uint64_t(first[index]) < right;
        }
        trim(result); return result;
    }
    static std::pair<int, Limbs> integer(const Value& value) {
        if (value.is_int()) {
            const auto number = value.as_int();
            const auto magnitude = number < 0 ? std::uint64_t(-(number + 1)) + 1 : std::uint64_t(number);
            return {number < 0 ? -1 : 1, words(magnitude)};
        }
        if (value.kind() != Value::Kind::BigInt)
            throw std::invalid_argument("NATIVE_CUSA_NON_DYADIC_PROGRESS");
        const auto& text = value.as_bigint();
        std::size_t position = !text.empty() && text[0] == '-' ? 1 : 0;
        const int sign = position ? -1 : 1;
        if (position == text.size()) throw std::invalid_argument("Invalid native integer");
        Limbs result;
        for (; position < text.size(); ++position) {
            if (text[position] < '0' || text[position] > '9') throw std::invalid_argument("Invalid native integer");
            std::uint64_t carry = static_cast<unsigned>(text[position] - '0');
            for (auto& word : result) {
                const std::uint64_t next = std::uint64_t(word) * 10 + carry;
                word = static_cast<std::uint32_t>(next); carry = next >> 32;
            }
            if (carry) result.push_back(static_cast<std::uint32_t>(carry));
        }
        trim(result); return {sign, std::move(result)};
    }
    static int power_of_two(const Limbs& value) {
        int answer = -1;
        for (std::size_t index = 0; index < value.size(); ++index) if (value[index]) {
            if (answer != -1 || (value[index] & (value[index] - 1)) != 0)
                throw std::invalid_argument("NATIVE_CUSA_NON_DYADIC_PROGRESS");
            const auto bit = index * 32 + trailing_zeros(value[index]);
            if (bit > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("NATIVE_CUSA_NON_DYADIC_PROGRESS");
            answer = static_cast<int>(bit);
        }
        if (answer == -1) throw std::invalid_argument("NATIVE_CUSA_NON_DYADIC_PROGRESS");
        return answer;
    }
    static Value integer_value(int sign, Limbs value) {
        if (value.empty()) return 0;
        std::vector<std::uint32_t> chunks;
        while (!value.empty()) {
            std::uint64_t remainder = 0;
            for (std::size_t index = value.size(); index != 0;) {
                --index;
                const auto current = (remainder << 32) | value[index];
                value[index] = static_cast<std::uint32_t>(current / 1000000000);
                remainder = current % 1000000000;
            }
            chunks.push_back(static_cast<std::uint32_t>(remainder)); trim(value);
        }
        std::string text = sign < 0 ? "-" : "";
        text += std::to_string(chunks.back());
        for (std::size_t index = chunks.size() - 1; index != 0;) {
            const auto digits = std::to_string(chunks[--index]);
            text.append(9 - digits.size(), '0'); text += digits;
        }
        std::int64_t small;
        const auto converted = std::from_chars(text.data(), text.data() + text.size(), small);
        if (converted.ec == std::errc{} && converted.ptr == text.data() + text.size()) return small;
        return Value::big_integer(std::move(text));
    }
    bool bit(std::size_t index) const {
        return index / 32 < coefficient_.size() && ((coefficient_[index / 32] >> (index % 32)) & 1);
    }
    bool any_below(std::size_t bits) const {
        const auto whole = std::min(bits / 32, coefficient_.size());
        for (std::size_t index = 0; index < whole; ++index) if (coefficient_[index]) return true;
        const unsigned partial = static_cast<unsigned>(bits % 32);
        return whole < coefficient_.size() && partial &&
            (coefficient_[whole] & ((std::uint32_t(1) << partial) - 1));
    }
    std::uint64_t rounded_shift(std::int64_t shift) const {
        if (shift <= 0) {
            const auto low = std::uint64_t(coefficient_[0]) |
                (coefficient_.size() > 1 ? std::uint64_t(coefficient_[1]) << 32 : 0);
            return low << static_cast<unsigned>(-shift);
        }
        const auto word = static_cast<std::size_t>(shift / 32);
        const unsigned offset = static_cast<unsigned>(shift % 32);
        std::uint64_t result = 0;
        if (word < coefficient_.size()) {
            const auto low = std::uint64_t(coefficient_[word]) |
                (word + 1 < coefficient_.size() ? std::uint64_t(coefficient_[word + 1]) << 32 : 0);
            result = low >> offset;
            if (offset && word + 2 < coefficient_.size())
                result |= std::uint64_t(coefficient_[word + 2]) << (64 - offset);
        }
        if (bit(static_cast<std::size_t>(shift - 1)) &&
                (any_below(static_cast<std::size_t>(shift - 1)) || (result & 1))) ++result;
        return result;
    }

    int sign_ = 0;
    Limbs coefficient_;
    int exponent_ = 0;
};

inline bool dyadic_pair_supported(const Value& pair) {
    try { (void)Dyadic::from_pair(pair); return true; }
    catch (const std::invalid_argument&) { return false; }
    catch (const std::overflow_error&) { return false; }
}

// CPython 3.12 fsum's non-overlapping partials algorithm and final tie correction.
// Reference: python/cpython Modules/mathmodule.c (Python Software Foundation License).
inline double dyadic_fsum(const std::vector<double>& values) {
    std::vector<double> partials;
    partials.reserve(std::min<std::size_t>(values.size(), 32));
    double special = 0., infinities = 0.;
    for (double incoming : values) {
        double running = incoming;
        std::size_t retained = 0;
        for (double previous : partials) {
            if (std::abs(running) < std::abs(previous)) std::swap(running, previous);
            const double combined = running + previous;
            const double rounded = combined - running;
            const double residual = previous - rounded;
            if (residual != 0.) partials[retained++] = residual;
            running = combined;
        }
        partials.resize(retained);
        if (running == 0.) continue;
        if (std::isfinite(running)) partials.push_back(running);
        else {
            if (std::isfinite(incoming)) throw std::overflow_error("intermediate overflow in fsum");
            if (std::isinf(incoming)) infinities += incoming;
            special += incoming;
            partials.clear();
        }
    }
    if (special != 0.) {
        if (std::isnan(infinities)) throw std::invalid_argument("-inf + inf in fsum");
        return special;
    }
    if (partials.empty()) return 0.;
    double result = partials.back();
    partials.pop_back();
    double residual = 0.;
    while (!partials.empty()) {
        const double previous = partials.back();
        partials.pop_back();
        const double combined = result + previous;
        const double rounded = combined - result;
        residual = previous - rounded;
        result = combined;
        if (residual != 0.) break;
    }
    if (!partials.empty() && ((residual < 0. && partials.back() < 0.) ||
            (residual > 0. && partials.back() > 0.))) {
        const double correction = residual * 2.;
        const double corrected = result + correction;
        if (corrected - result == correction) result = corrected;
    }
    return result;
}

}  // namespace resectionlab::sim
