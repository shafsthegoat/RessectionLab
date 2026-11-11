#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace resectionlab::sim {

class Value {
    struct BigInteger { std::string decimal; };

public:
    using Array = std::vector<Value>;
    using Object = std::vector<std::pair<std::string, Value>>;
    enum class Kind { Null, Bool, Int, Double, String, Array, Object, BigInt };

    Value() noexcept = default;
    Value(std::nullptr_t) noexcept : data_(std::monostate{}) {}
    Value(bool value) : data_(value) {}
    template <class Integer, std::enable_if_t<std::is_integral_v<Integer> &&
        !std::is_same_v<std::remove_cv_t<Integer>, bool>, int> = 0>
    Value(Integer value) : data_(checked_integer(value)) {}
    Value(double value) : data_(value) {}
    Value(std::string value) : data_(std::move(value)) {}
    Value(std::string_view value) : data_(std::string(value)) {}
    Value(const char* value) : data_(checked_string(value)) {}
    Value(Array value) : data_(std::move(value)) {}
    Value(Object value) : data_(std::move(value)) {}

    static Value array(Array values = {}) { return Value(std::move(values)); }
    static Value object(Object values = {}) { return Value(std::move(values)); }
    static Value big_integer(std::string decimal) {
        const std::size_t first = !decimal.empty() && decimal.front() == '-' ? 1 : 0;
        if (first == decimal.size() ||
                (decimal[first] == '0' && (first != 0 || decimal.size() != 1))) {
            throw std::invalid_argument("Big integer requires canonical decimal digits");
        }
        for (std::size_t index = first; index < decimal.size(); ++index) {
            if (decimal[index] < '0' || decimal[index] > '9') {
                throw std::invalid_argument("Big integer requires canonical decimal digits");
            }
        }
        Value result;
        result.data_ = BigInteger{std::move(decimal)};
        return result;
    }

    Kind kind() const noexcept { return static_cast<Kind>(data_.index()); }
    bool is(Kind value) const noexcept { return kind() == value; }
    bool is_null() const noexcept { return is(Kind::Null); }
    bool is_bool() const noexcept { return is(Kind::Bool); }
    bool is_int() const noexcept { return is(Kind::Int); }
    bool is_double() const noexcept { return is(Kind::Double); }
    bool is_number() const noexcept { return is_int() || is_double(); }
    bool is_string() const noexcept { return is(Kind::String); }
    bool is_array() const noexcept { return is(Kind::Array); }
    bool is_object() const noexcept { return is(Kind::Object); }
    bool is_bigint() const noexcept { return is(Kind::BigInt); }

    std::nullptr_t as_null() const {
        if (!is_null()) throw std::bad_variant_access();
        return nullptr;
    }
    bool& as_bool() { return std::get<bool>(data_); }
    bool as_bool() const { return std::get<bool>(data_); }
    std::int64_t& as_int() { return std::get<std::int64_t>(data_); }
    std::int64_t as_int() const { return std::get<std::int64_t>(data_); }
    double& as_double() { return std::get<double>(data_); }
    double as_double() const { return std::get<double>(data_); }
    double as_number() const { return is_int() ? static_cast<double>(as_int()) : as_double(); }
    std::string& as_string() { return std::get<std::string>(data_); }
    const std::string& as_string() const { return std::get<std::string>(data_); }
    Array& as_array() { return std::get<Array>(data_); }
    const Array& as_array() const { return std::get<Array>(data_); }
    Object& as_object() { return std::get<Object>(data_); }
    const Object& as_object() const { return std::get<Object>(data_); }
    const std::string& as_bigint() const { return std::get<BigInteger>(data_).decimal; }

    bool truthy() const {
        switch (kind()) {
            case Kind::Null: return false;
            case Kind::Bool: return as_bool();
            case Kind::Int: return as_int() != 0;
            case Kind::Double: return as_double() != 0.;
            case Kind::String: return !as_string().empty();
            case Kind::Array: return !as_array().empty();
            case Kind::Object: return !as_object().empty();
            case Kind::BigInt: return as_bigint() != "0";
        }
        throw std::bad_variant_access();
    }

    Value* find(std::string_view key) noexcept {
        auto* members = std::get_if<Object>(&data_);
        if (members != nullptr) {
            for (auto& member : *members) {
                if (std::string_view(member.first) == key) return &member.second;
            }
        }
        return nullptr;
    }
    const Value* find(std::string_view key) const noexcept {
        const auto* members = std::get_if<Object>(&data_);
        if (members != nullptr) {
            for (const auto& member : *members) {
                if (std::string_view(member.first) == key) return &member.second;
            }
        }
        return nullptr;
    }
    bool contains(std::string_view key) const noexcept { return find(key) != nullptr; }
    Value& at(std::string_view key) {
        auto* value = find(key);
        if (value == nullptr) throw std::out_of_range(std::string(key));
        return *value;
    }
    const Value& at(std::string_view key) const {
        const auto* value = find(key);
        if (value == nullptr) throw std::out_of_range(std::string(key));
        return *value;
    }
    Value& at(std::size_t index) { return as_array().at(index); }
    const Value& at(std::size_t index) const { return as_array().at(index); }

    Value& set(std::string key, Value value) {
        auto& members = as_object();
        for (auto& member : members) {
            if (member.first == key) {
                member.second = std::move(value);
                return member.second;
            }
        }
        members.emplace_back(std::move(key), std::move(value));
        return members.back().second;
    }
    bool erase(std::string_view key) {
        auto& members = as_object();
        for (auto member = members.begin(); member != members.end(); ++member) {
            if (std::string_view(member->first) == key) {
                members.erase(member);
                return true;
            }
        }
        return false;
    }
    Value& append(Value value) {
        auto& values = as_array();
        values.push_back(std::move(value));
        return values.back();
    }

private:
    template <class Integer>
    static std::int64_t checked_integer(Integer value) {
        if constexpr (std::numeric_limits<Integer>::digits > std::numeric_limits<std::int64_t>::digits) {
            if (value > static_cast<Integer>(std::numeric_limits<std::int64_t>::max())) {
                throw std::out_of_range("JSON integer exceeds int64 range");
            }
            if constexpr (std::is_signed_v<Integer>) {
                if (value < static_cast<Integer>(std::numeric_limits<std::int64_t>::min())) {
                    throw std::out_of_range("JSON integer exceeds int64 range");
                }
            }
        }
        return static_cast<std::int64_t>(value);
    }
    static std::string checked_string(const char* value) {
        if (value == nullptr) throw std::invalid_argument("JSON string pointer is null");
        return std::string(value);
    }

    std::variant<std::monostate, bool, std::int64_t, double, std::string, Array, Object, BigInteger> data_;
};

using Array = Value::Array;
using Object = Value::Object;

}  // namespace resectionlab::sim
