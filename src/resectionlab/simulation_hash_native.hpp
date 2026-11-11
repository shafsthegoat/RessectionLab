#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif
#include "simulation_value_native.hpp"

namespace resectionlab::sim {

#if defined(__APPLE__)
namespace hash_detail {
// Weak-link the public libc++ overload so older deployment targets remain loadable.
extern "C" std::to_chars_result apple_to_chars(char*, char*, double, std::chars_format)
    __asm__("__ZNSt3__18to_charsEPcS0_dNS_12chars_formatE") __attribute__((weak_import));
}
#endif

inline bool native_hash_supported() noexcept {
#if defined(__APPLE__)
    if (__builtin_available(macOS 13.3, *)) return hash_detail::apple_to_chars != nullptr;
    return false;
#else
    return true;
#endif
}
namespace hash_detail {

inline void append_double(std::string& output, double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("metadata cannot contain nonfinite numbers");
    char buffer[64];
    std::to_chars_result formatted;
#if defined(__APPLE__)
    if (native_hash_supported())
        formatted = apple_to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    else throw std::invalid_argument("Native simulation number formatting requires macOS 13.3");
#else
    formatted = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
#endif
    if (formatted.ec != std::errc{}) throw std::runtime_error("Could not format JSON number");
    const char* cursor = buffer;
    if (*cursor == '-') { output.push_back('-'); ++cursor; }
    char digits[32];
    std::size_t count = 0;
    while (cursor != formatted.ptr && *cursor != 'e') {
        if (*cursor != '.') digits[count++] = *cursor;
        ++cursor;
    }
    if (cursor == formatted.ptr || count == 0) throw std::runtime_error("Invalid scientific number");
    ++cursor;
    const bool negative_exponent = *cursor == '-';
    if (*cursor == '+' || *cursor == '-') ++cursor;
    int exponent = 0;
    while (cursor != formatted.ptr) exponent = exponent * 10 + (*cursor++ - '0');
    if (negative_exponent) exponent = -exponent;

    // Python repr uses the same shortest digits, with its fixed notation thresholds.
    if (exponent >= -4 && exponent < 16) {
        const int point = exponent + 1;
        if (point <= 0) {
            output.append("0.");
            output.append(static_cast<std::size_t>(-point), '0');
            output.append(digits, count);
        } else if (static_cast<std::size_t>(point) >= count) {
            output.append(digits, count);
            output.append(static_cast<std::size_t>(point) - count, '0');
            output.append(".0");
        } else {
            output.append(digits, static_cast<std::size_t>(point));
            output.push_back('.');
            output.append(digits + point, count - static_cast<std::size_t>(point));
        }
        return;
    }
    output.push_back(digits[0]);
    if (count > 1) {
        output.push_back('.');
        output.append(digits + 1, count - 1);
    }
    output.push_back('e');
    output.push_back(exponent < 0 ? '-' : '+');
    const unsigned magnitude = static_cast<unsigned>(exponent < 0 ? -exponent : exponent);
    if (magnitude < 10) output.push_back('0');
    const auto exponent_text = std::to_chars(buffer, buffer + sizeof(buffer), magnitude);
    output.append(buffer, static_cast<std::size_t>(exponent_text.ptr - buffer));
}

inline std::uint32_t next_codepoint(std::string_view text, std::size_t& position) {
    const auto first = static_cast<unsigned char>(text[position++]);
    if (first < 0x80) return first;
    unsigned trailing;
    std::uint32_t codepoint;
    std::uint32_t minimum;
    if (first >= 0xc2 && first <= 0xdf) {
        trailing = 1; codepoint = first & 0x1f; minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
        trailing = 2; codepoint = first & 0x0f; minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
        trailing = 3; codepoint = first & 7; minimum = 0x10000;
    } else throw std::invalid_argument("JSON string is not UTF-8");
    if (trailing > text.size() - position) throw std::invalid_argument("JSON string is not UTF-8");
    for (unsigned index = 0; index < trailing; ++index) {
        const auto byte = static_cast<unsigned char>(text[position++]);
        if ((byte & 0xc0) != 0x80) throw std::invalid_argument("JSON string is not UTF-8");
        codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    // Surrogate-pass UTF-8 retains lone Python surrogates as separate codepoints.
    if (codepoint < minimum || codepoint > 0x10ffff)
        throw std::invalid_argument("JSON string is not UTF-8");
    return codepoint;
}

inline void append_escaped(std::string& output, std::uint32_t codepoint) {
    constexpr char hex[] = "0123456789abcdef";
    const char encoded[] = {'\\', 'u', hex[(codepoint >> 12) & 15],
        hex[(codepoint >> 8) & 15], hex[(codepoint >> 4) & 15], hex[codepoint & 15]};
    output.append(encoded, sizeof(encoded));
}

inline void append_string(std::string& output, std::string_view value) {
    output.push_back('"');
    std::size_t position = 0;
    while (position < value.size()) {
        const std::size_t start = position;
        while (position < value.size()) {
            const auto byte = static_cast<unsigned char>(value[position]);
            if (byte < 0x20 || byte > 0x7e || byte == '"' || byte == '\\') break;
            ++position;
        }
        output.append(value.data() + start, position - start);
        if (position == value.size()) break;
        std::uint32_t codepoint = next_codepoint(value, position);
        switch (codepoint) {
            case '"': output.append("\\\""); break;
            case '\\': output.append("\\\\"); break;
            case '\b': output.append("\\b"); break;
            case '\f': output.append("\\f"); break;
            case '\n': output.append("\\n"); break;
            case '\r': output.append("\\r"); break;
            case '\t': output.append("\\t"); break;
            default:
                if (codepoint <= 0xffff) append_escaped(output, codepoint);
                else {
                    codepoint -= 0x10000;
                    append_escaped(output, 0xd800 + (codepoint >> 10));
                    append_escaped(output, 0xdc00 + (codepoint & 0x3ff));
                }
        }
    }
    output.push_back('"');
}

inline void append_json(std::string& output, const Value& value) {
    switch (value.kind()) {
        case Value::Kind::Null: output.append("null"); return;
        case Value::Kind::Bool: output.append(value.as_bool() ? "true" : "false"); return;
        case Value::Kind::Int: {
            char buffer[32];
            const auto formatted = std::to_chars(buffer, buffer + sizeof(buffer), value.as_int());
            output.append(buffer, static_cast<std::size_t>(formatted.ptr - buffer));
            return;
        }
        case Value::Kind::Double: append_double(output, value.as_double()); return;
        case Value::Kind::BigInt: output.append(value.as_bigint()); return;
        case Value::Kind::String: append_string(output, value.as_string()); return;
        case Value::Kind::Array: {
            output.push_back('[');
            bool first = true;
            for (const auto& item : value.as_array()) {
                if (!first) output.push_back(',');
                append_json(output, item);
                first = false;
            }
            output.push_back(']');
            return;
        }
        case Value::Kind::Object: {
            std::vector<const Object::value_type*> ordered;
            ordered.reserve(value.as_object().size());
            for (const auto& item : value.as_object()) ordered.push_back(&item);
            // UTF-8 byte order equals Unicode codepoint order, including surrogate pass.
            std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
                return left->first < right->first;
            });
            output.push_back('{');
            const std::string* previous = nullptr;
            for (const auto* item : ordered) {
                if (previous != nullptr) {
                    if (*previous == item->first) throw std::invalid_argument("JSON object has duplicate keys");
                    output.push_back(',');
                }
                append_string(output, item->first);
                output.push_back(':');
                append_json(output, item->second);
                previous = &item->first;
            }
            output.push_back('}');
            return;
        }
    }
    throw std::invalid_argument("Unknown JSON value kind");
}

#if !defined(__APPLE__)
inline std::array<unsigned char, 32> sha256(std::string_view payload) {
    constexpr std::uint32_t constants[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t, 8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const auto rotate = [](std::uint32_t word, unsigned count) {
        return (word >> count) | (word << (32 - count));
    };
    const auto compress = [&](const unsigned char* block) {
        std::uint32_t words[64];
        for (unsigned index = 0; index < 16; ++index) {
            words[index] = (std::uint32_t(block[4*index]) << 24) |
                (std::uint32_t(block[4*index+1]) << 16) |
                (std::uint32_t(block[4*index+2]) << 8) | block[4*index+3];
        }
        for (unsigned index = 16; index < 64; ++index) {
            const auto left = words[index-15], right = words[index-2];
            const auto s0 = rotate(left,7) ^ rotate(left,18) ^ (left >> 3);
            const auto s1 = rotate(right,17) ^ rotate(right,19) ^ (right >> 10);
            words[index] = words[index-16] + s0 + words[index-7] + s1;
        }
        auto a=state[0], b=state[1], c=state[2], d=state[3];
        auto e=state[4], f=state[5], g=state[6], h=state[7];
        for (unsigned index = 0; index < 64; ++index) {
            const auto s1 = rotate(e,6) ^ rotate(e,11) ^ rotate(e,25);
            const auto choice = (e & f) ^ (~e & g);
            const auto first = h + s1 + choice + constants[index] + words[index];
            const auto s0 = rotate(a,2) ^ rotate(a,13) ^ rotate(a,22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto second = s0 + majority;
            h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    };
    if (payload.size() > std::numeric_limits<std::uint64_t>::max() / 8)
        throw std::length_error("SHA256 input is too large");
    const auto* data = reinterpret_cast<const unsigned char*>(payload.data());
    std::size_t position = 0;
    while (payload.size() - position >= 64) { compress(data + position); position += 64; }
    unsigned char tail[128]{};
    const std::size_t remaining = payload.size() - position;
    if (remaining != 0) std::memcpy(tail, data + position, remaining);
    tail[remaining] = 0x80;
    const std::size_t padded = remaining < 56 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(payload.size()) * 8;
    for (unsigned index = 0; index < 8; ++index)
        tail[padded - index - 1] = static_cast<unsigned char>(bits >> (8 * index));
    compress(tail);
    if (padded == 128) compress(tail + 64);
    std::array<unsigned char, 32> result{};
    for (unsigned index = 0; index < 32; ++index)
        result[index] = static_cast<unsigned char>(state[index/4] >> (24 - 8*(index%4)));
    return result;
}
#else
inline std::array<unsigned char, 32> sha256(std::string_view payload) {
    std::array<unsigned char, 32> result{};
    if (payload.size() <= std::numeric_limits<CC_LONG>::max()) {
        CC_SHA256(payload.data(), static_cast<CC_LONG>(payload.size()), result.data());
        return result;
    }
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::size_t position = 0;
    while (position < payload.size()) {
        const auto count = std::min(payload.size() - position,
            static_cast<std::size_t>(std::numeric_limits<CC_LONG>::max()));
        CC_SHA256_Update(&context, payload.data() + position, static_cast<CC_LONG>(count));
        position += count;
    }
    CC_SHA256_Final(result.data(), &context);
    return result;
}
#endif

}  // namespace hash_detail

inline std::string canonical_json(const Value& value) {
    std::string result;
    result.reserve(512);
    hash_detail::append_json(result, value);
    return result;
}

inline std::string digest(const Value& value) {
    const auto bytes = hash_detail::sha256(canonical_json(value));
    constexpr char hex[] = "0123456789abcdef";
    std::string result("sha256:");
    result.reserve(71);
    for (const auto byte : bytes) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}

}  // namespace resectionlab::sim
