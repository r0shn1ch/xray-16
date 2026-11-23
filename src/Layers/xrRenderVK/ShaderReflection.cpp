#include "ShaderReflection.h"
#include "ShaderModule.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t Missing = ~0u;
struct Type
{
    uint32_t opcode{}, element{}, sampled{}, length_id{}, width{}, signedness{}, count{}, stride{Missing};
    uint32_t image_dimension{Missing}, image_arrayed{}, image_multisampled{};
    std::vector<uint32_t> members;
    bool buffer_block{};
};
struct Variable
{
    uint32_t type{}, storage{};
    bool present{};
};
struct ConstantRecord
{
    uint32_t type{}, operation{}, specialization_id{Missing};
    uint64_t value{};
    std::vector<uint32_t> operands;
    bool present{};
};

uint64_t width_mask(uint32_t width)
{
    return width >= 64 ? UINT64_MAX : width ? ((uint64_t(1) << width) - 1) : 0;
}

int64_t signed_value(uint64_t value, uint32_t width)
{
    const uint64_t mask = width_mask(width);
    value &= mask;
    if (width < 64 && width && (value & (uint64_t(1) << (width - 1))))
        value |= ~mask;
    return static_cast<int64_t>(value);
}
}

bool resolve_shader_descriptor_counts(const ShaderReflection& reflection,
    const std::vector<ShaderSpecializationValue>& values,
    std::vector<uint32_t>& counts, std::string& error)
{
    std::unordered_map<uint32_t, const ShaderConstantExpression*> expressions;
    std::unordered_map<uint32_t, const ShaderConstantExpression*> specialization_ids;
    expressions.reserve(reflection.constants.size());
    specialization_ids.reserve(reflection.constants.size());
    for (const ShaderConstantExpression& expression : reflection.constants)
    {
        if (!expression.result_id || !expressions.emplace(expression.result_id, &expression).second)
        {
            error = "SPIR-V reflection: duplicate specialization expression ID";
            return false;
        }
        if (expression.specialization_id != Missing &&
            !specialization_ids.emplace(expression.specialization_id, &expression).second)
        {
            error = "SPIR-V reflection: duplicate specialization constant ID";
            return false;
        }
    }

    std::unordered_map<uint32_t, uint64_t> overrides;
    overrides.reserve(values.size());
    for (const ShaderSpecializationValue& value : values)
    {
        const auto found = specialization_ids.find(value.constant_id);
        const uint32_t width = found == specialization_ids.end() ? 0 : found->second->bit_width;
        const uint32_t expected_size = width == 1 ? 4 : width / 8;
        if (found != specialization_ids.end() &&
            (!width || width > 64 || (width != 1 && width % 8) || value.byte_size != expected_size))
        {
            error = "SPIR-V reflection: specialization value size does not match its scalar type";
            return false;
        }
        if (!overrides.emplace(value.constant_id,
                width ? value.value & width_mask(width) : value.value).second)
        {
            error = "SPIR-V reflection: duplicate specialization value";
            return false;
        }
    }

    std::unordered_map<uint32_t, uint64_t> evaluated;
    std::unordered_map<uint32_t, bool> visiting;
    std::function<std::optional<uint64_t>(uint32_t, uint32_t)> evaluate;
    evaluate = [&](uint32_t id, uint32_t depth) -> std::optional<uint64_t> {
        if (depth >= 1024)
            return std::nullopt;
        if (const auto cached = evaluated.find(id); cached != evaluated.end())
            return cached->second;
        const auto found = expressions.find(id);
        if (found == expressions.end() || visiting[id])
            return std::nullopt;
        const ShaderConstantExpression& expression = *found->second;
        const uint32_t width = expression.bit_width;
        if (!width || width > 64)
            return std::nullopt;
        if (expression.specialization_id != Missing)
        {
            const auto value = overrides.find(expression.specialization_id);
            if (value != overrides.end())
            {
                const uint64_t result = value->second & width_mask(width);
                evaluated.emplace(id, result);
                return result;
            }
        }
        if (!expression.operation)
        {
            const uint64_t result = expression.default_value & width_mask(width);
            evaluated.emplace(id, result);
            return result;
        }

        visiting[id] = true;
        std::vector<uint64_t> operands;
        operands.reserve(expression.operands.size());
        for (uint32_t operand : expression.operands)
        {
            const auto value = evaluate(operand, depth + 1);
            if (!value)
            {
                visiting[id] = false;
                return std::nullopt;
            }
            operands.push_back(*value);
        }
        const auto unary = [&]() { return operands.size() == 1; };
        const auto binary = [&]() { return operands.size() == 2; };
        const auto logical = [](uint64_t value) { return value != 0; };
        std::optional<uint64_t> result;
        switch (expression.operation)
        {
        case 83: // OpCopyObject
            if (unary()) result = operands[0];
            break;
        case 113: // OpUConvert
        case 124: // OpBitcast
            if (unary()) result = operands[0];
            break;
        case 114: // OpSConvert
            if (unary()) result = static_cast<uint64_t>(signed_value(
                operands[0], expressions.at(expression.operands[0])->bit_width));
            break;
        case 126: // OpSNegate
            if (unary()) result = uint64_t(0) - operands[0];
            break;
        case 128: // OpIAdd
            if (binary()) result = operands[0] + operands[1];
            break;
        case 130: // OpISub
            if (binary()) result = operands[0] - operands[1];
            break;
        case 132: // OpIMul
            if (binary()) result = operands[0] * operands[1];
            break;
        case 134: // OpUDiv
            if (binary() && operands[1]) result = operands[0] / operands[1];
            break;
        case 135: // OpSDiv
            if (binary() && operands[1])
            {
                const int64_t lhs = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width);
                const int64_t rhs = signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
                if (!(lhs == std::numeric_limits<int64_t>::min() && rhs == -1))
                    result = static_cast<uint64_t>(lhs / rhs);
            }
            break;
        case 137: // OpUMod
            if (binary() && operands[1]) result = operands[0] % operands[1];
            break;
        case 138: // OpSRem
        case 139: // OpSMod
            if (binary() && operands[1])
            {
                const int64_t lhs = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width);
                const int64_t rhs = signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
                if (!(lhs == std::numeric_limits<int64_t>::min() && rhs == -1))
                {
                    int64_t remainder = lhs % rhs;
                    if (expression.operation == 139 && remainder && ((remainder < 0) != (rhs < 0)))
                        remainder += rhs;
                    result = static_cast<uint64_t>(remainder);
                }
            }
            break;
        case 194: // OpShiftRightLogical
            if (binary() && operands[1] < width) result = operands[0] >> operands[1];
            break;
        case 195: // OpShiftRightArithmetic
            if (binary() && operands[1] < width)
                result = static_cast<uint64_t>(signed_value(operands[0], width) >> operands[1]);
            break;
        case 196: // OpShiftLeftLogical
            if (binary() && operands[1] < width) result = operands[0] << operands[1];
            break;
        case 197: // OpBitwiseOr
            if (binary()) result = operands[0] | operands[1];
            break;
        case 198: // OpBitwiseXor
            if (binary()) result = operands[0] ^ operands[1];
            break;
        case 199: // OpBitwiseAnd
            if (binary()) result = operands[0] & operands[1];
            break;
        case 200: // OpNot
            if (unary()) result = ~operands[0];
            break;
        case 164: // OpLogicalEqual
            if (binary()) result = logical(operands[0]) == logical(operands[1]);
            break;
        case 165: // OpLogicalNotEqual
            if (binary()) result = logical(operands[0]) != logical(operands[1]);
            break;
        case 166: // OpLogicalOr
            if (binary()) result = logical(operands[0]) || logical(operands[1]);
            break;
        case 167: // OpLogicalAnd
            if (binary()) result = logical(operands[0]) && logical(operands[1]);
            break;
        case 168: // OpLogicalNot
            if (unary()) result = !logical(operands[0]);
            break;
        case 169: // OpSelect
            if (operands.size() == 3) result = logical(operands[0]) ? operands[1] : operands[2];
            break;
        case 170: // OpIEqual
            if (binary()) result = operands[0] == operands[1];
            break;
        case 171: // OpINotEqual
            if (binary()) result = operands[0] != operands[1];
            break;
        case 172: // OpUGreaterThan
            if (binary()) result = operands[0] > operands[1];
            break;
        case 173: // OpSGreaterThan
            if (binary()) result = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width) >
                signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
            break;
        case 174: // OpUGreaterThanEqual
            if (binary()) result = operands[0] >= operands[1];
            break;
        case 175: // OpSGreaterThanEqual
            if (binary()) result = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width) >=
                signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
            break;
        case 176: // OpULessThan
            if (binary()) result = operands[0] < operands[1];
            break;
        case 177: // OpSLessThan
            if (binary()) result = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width) <
                signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
            break;
        case 178: // OpULessThanEqual
            if (binary()) result = operands[0] <= operands[1];
            break;
        case 179: // OpSLessThanEqual
            if (binary()) result = signed_value(operands[0], expressions.at(expression.operands[0])->bit_width) <=
                signed_value(operands[1], expressions.at(expression.operands[1])->bit_width);
            break;
        default:
            break;
        }
        visiting[id] = false;
        if (!result)
            return std::nullopt;
        const uint64_t masked = *result & width_mask(width);
        evaluated.emplace(id, masked);
        return masked;
    };

    std::vector<uint32_t> resolved(reflection.descriptors.size(), 1);
    for (size_t descriptor_index = 0; descriptor_index < reflection.descriptors.size(); ++descriptor_index)
    {
        const ShaderDescriptor& descriptor = reflection.descriptors[descriptor_index];
        if (descriptor.runtime_sized)
        {
            resolved[descriptor_index] = 0;
            continue;
        }
        uint64_t count = 1;
        for (uint32_t length_id : descriptor.array_length_ids)
        {
            const auto length = evaluate(length_id, 0);
            if (!length || !*length || *length > UINT32_MAX || count > UINT32_MAX / *length)
            {
                error = "SPIR-V reflection: descriptor array length could not be resolved";
                return false;
            }
            count *= *length;
        }
        resolved[descriptor_index] = static_cast<uint32_t>(count);
    }
    counts = std::move(resolved);
    error.clear();
    return true;
}

bool reflect_spirv_resources(const void* bytes, size_t size, uint32_t stage,
    const char* entry, ShaderReflection& result, std::string& error)
{
    result = {};
    if (!has_spirv_entry(bytes, size, stage, entry))
    {
        error = "SPIR-V reflection: missing or malformed entry point";
        return false;
    }
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    std::memcpy(words.data(), bytes, size);
    const uint32_t bound = words[3];
    if (bound > (1u << 20))
    {
        error = "SPIR-V reflection: excessive ID bound";
        return false;
    }
    std::vector<Type> types(bound);
    std::vector<Variable> variables(bound);
    std::vector<uint32_t> sets(bound, Missing), bindings(bound, Missing);
    std::vector<uint32_t> locations(bound, Missing), components(bound);
    std::vector<bool> entry_interfaces(bound), builtins(bound);
    std::vector<uint32_t> constants(bound, Missing);
    std::vector<std::vector<uint32_t>> member_offsets(bound), member_matrix_strides(bound);
    std::vector<std::vector<uint32_t>> member_locations(bound), member_components(bound);
    std::vector<std::vector<bool>> member_builtins(bound);
    const auto id_valid = [bound](uint32_t id) { return id && id < bound; };
    std::vector<uint32_t> specialization_ids(bound, Missing);
    std::vector<ConstantRecord> constant_records(bound);
    for (size_t offset = 5; offset < words.size();)
    {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        if (!count || count > words.size() - offset)
        {
            error = "SPIR-V reflection: malformed instruction length";
            return false;
        }
        const auto word = [&](uint32_t index) { return words[offset + index]; };
        if (opcode == 15 && count >= 4 && word(1) == stage) // OpEntryPoint
        {
            const char* name = reinterpret_cast<const char*>(words.data() + offset + 3);
            const size_t capacity = (count - 3) * sizeof(uint32_t);
            const auto* end = static_cast<const char*>(std::memchr(name, 0, capacity));
            if (end && size_t(end - name) == std::strlen(entry) &&
                std::memcmp(name, entry, std::strlen(entry)) == 0)
            {
                const size_t name_words = (size_t(end - name) + 4) / 4;
                for (size_t index = 3 + name_words; index < count; ++index)
                    if (id_valid(word(uint32_t(index)))) entry_interfaces[word(uint32_t(index))] = true;
            }
        }
        else if (opcode == 71) // OpDecorate
        {
            if (count < 3 || !id_valid(word(1)) ||
                ((word(2) == 1 || word(2) == 6 || word(2) == 30 || word(2) == 31 ||
                    word(2) == 33 || word(2) == 34) && count < 4))
            {
                error = "SPIR-V reflection: malformed decoration";
                return false;
            }
            if (word(2) == 1) specialization_ids[word(1)] = word(3); // SpecId
            if (word(2) == 33) bindings[word(1)] = word(3); // Binding
            if (word(2) == 34) sets[word(1)] = word(3); // DescriptorSet
            if (word(2) == 3) types[word(1)].buffer_block = true; // BufferBlock
            if (word(2) == 6) types[word(1)].stride = word(3); // ArrayStride
            if (word(2) == 11) builtins[word(1)] = true; // BuiltIn
            if (word(2) == 30) locations[word(1)] = word(3); // Location
            if (word(2) == 31) components[word(1)] = word(3); // Component
        }
        else if (opcode == 72) // OpMemberDecorate
        {
            if (count < 4 || !id_valid(word(1)) ||
                ((word(3) == 35 || word(3) == 7 || word(3) == 30 ||
                    word(3) == 31 || word(3) == 11) && count < 5) || word(2) > 4096)
            {
                error = "SPIR-V reflection: malformed member decoration";
                return false;
            }
            if (word(3) == 35 || word(3) == 7)
            {
                auto& values = word(3) == 35 ? member_offsets[word(1)] : member_matrix_strides[word(1)];
                if (values.size() <= word(2)) values.resize(word(2) + 1, Missing);
                values[word(2)] = word(4);
            }
            if (word(3) == 30 || word(3) == 31)
            {
                auto& values = word(3) == 30 ? member_locations[word(1)] : member_components[word(1)];
                if (values.size() <= word(2)) values.resize(word(2) + 1, Missing);
                values[word(2)] = word(4);
            }
            if (word(3) == 11)
            {
                auto& values = member_builtins[word(1)];
                if (values.size() <= word(2)) values.resize(word(2) + 1);
                values[word(2)] = true;
            }
        }
        else if (opcode == 20 || opcode == 21 || opcode == 22 || opcode == 23 ||
            opcode == 24 || opcode == 25 || opcode == 26 || opcode == 27 ||
            opcode == 28 || opcode == 29 || opcode == 30 || opcode == 32 || opcode == 5341)
        {
            const uint32_t minimum = opcode == 25 ? 9 : opcode == 28 || opcode == 32 ? 4 :
                opcode == 21 || opcode == 23 || opcode == 24 ? 4 :
                opcode == 22 || opcode == 27 || opcode == 29 ? 3 : 2;
            if (count < minimum || !id_valid(word(1)))
            {
                error = "SPIR-V reflection: malformed type instruction";
                return false;
            }
            Type& type = types[word(1)];
            type.opcode = opcode;
            if (opcode == 27 || opcode == 28 || opcode == 29)
                type.element = word(2);
            if (opcode == 28) type.length_id = word(3);
            if (opcode == 32) type.element = word(3);
            if (opcode == 25)
            {
                type.element = word(2);
                type.image_dimension = word(3);
                type.image_arrayed = word(5);
                type.image_multisampled = word(6);
                type.sampled = word(7);
            }
            if (opcode == 20) type.width = 1;
            if (opcode == 21 || opcode == 22) type.width = word(2);
            if (opcode == 21) type.signedness = word(3);
            if (opcode == 23 || opcode == 24)
            {
                type.element = word(2);
                type.count = word(3);
            }
            if (opcode == 30) type.members.assign(words.begin() + offset + 2,
                words.begin() + offset + count);
        }
        else if (opcode == 41 || opcode == 42 || opcode == 43 || opcode == 46 ||
            opcode == 48 || opcode == 49 || opcode == 50) // scalar constants
        {
            const bool boolean_constant = opcode == 41 || opcode == 42 || opcode == 46 ||
                opcode == 48 || opcode == 49;
            const bool specialization = opcode == 48 || opcode == 49 || opcode == 50;
            if (count < (boolean_constant ? 3u : 4u) || !id_valid(word(1)) || !id_valid(word(2)))
            {
                error = specialization ? "SPIR-V reflection: malformed specialization constant" :
                    "SPIR-V reflection: malformed constant";
                return false;
            }
            const uint32_t result_id = word(2);
            const uint32_t type_id = word(1);
            const uint32_t width = types[type_id].width;
            uint64_t value = opcode == 41 || opcode == 48 ? 1u : 0u;
            if (opcode == 43 || opcode == 50)
            {
                value = word(3);
                if (width > 32)
                {
                    if (count < 5)
                    {
                        error = "SPIR-V reflection: malformed 64-bit constant";
                        return false;
                    }
                    value |= uint64_t(word(4)) << 32;
                }
            }
            constant_records[result_id] = {type_id, 0, Missing, value, {}, true};
            if (!specialization)
                constants[result_id] = opcode == 43 && width > 32 ? Missing : static_cast<uint32_t>(value);
            else
                constants[result_id] = Missing;
        }
        else if (opcode == 52) // OpSpecConstantOp
        {
            if (count < 5 || !id_valid(word(1)) || !id_valid(word(2)))
            {
                error = "SPIR-V reflection: malformed specialization operation";
                return false;
            }
            ConstantRecord record{};
            record.type = word(1);
            record.operation = word(3);
            record.present = true;
            record.operands.assign(words.begin() + offset + 4, words.begin() + offset + count);
            constant_records[word(2)] = std::move(record);
            constants[word(2)] = Missing;
        }
        else if (opcode == 59) // OpVariable
        {
            if (count < 4 || !id_valid(word(1)) || !id_valid(word(2)))
            {
                error = "SPIR-V reflection: invalid variable ID";
                return false;
            }
            variables[word(2)] = {word(1), word(3), true};
        }
        offset += count;
    }
    for (uint32_t id = 1; id < bound; ++id)
    {
        const ConstantRecord& record = constant_records[id];
        if (!record.present || !id_valid(record.type))
            continue;
        const Type& type = types[record.type];
        if (type.opcode != 20 && type.opcode != 21)
            continue;
        ShaderConstantExpression expression;
        expression.result_id = id;
        expression.specialization_id = specialization_ids[id];
        expression.operation = record.operation;
        expression.bit_width = type.width;
        expression.signedness = type.signedness;
        expression.default_value = record.value;
        expression.operands = record.operands;
        result.constants.push_back(std::move(expression));
    }
    std::function<std::optional<uint64_t>(uint32_t, uint32_t, uint32_t)> size_of;
    size_of = [&](uint32_t type_id, uint32_t matrix_stride, uint32_t depth)
        -> std::optional<uint64_t> {
        if (!id_valid(type_id) || depth >= bound) return std::nullopt;
        const Type& type = types[type_id];
        if (type.opcode == 32) return size_of(type.element, matrix_stride, depth + 1);
        if (type.opcode == 20) return 4; // bool occupies four bytes in Vulkan blocks
        if (type.opcode == 21 || type.opcode == 22)
            return type.width && type.width % 8 == 0 ?
                std::optional<uint64_t>(type.width / 8) : std::nullopt;
        if (type.opcode == 23)
        {
            const auto scalar = size_of(type.element, Missing, depth + 1);
            return scalar && type.count && type.count <= 4 ?
                std::optional<uint64_t>(*scalar * type.count) : std::nullopt;
        }
        if (type.opcode == 24)
        {
            if (matrix_stride == Missing || !type.count) return std::nullopt;
            return uint64_t(matrix_stride) * type.count;
        }
        if (type.opcode == 28)
        {
            const uint32_t length = id_valid(type.length_id) ? constants[type.length_id] : Missing;
            if (!length || length == Missing || type.stride == Missing ||
                uint64_t(length) * type.stride > UINT32_MAX)
                return std::nullopt;
            return uint64_t(length) * type.stride;
        }
        if (type.opcode == 30)
        {
            uint64_t end = 0;
            for (size_t member = 0; member < type.members.size(); ++member)
            {
                if (member >= member_offsets[type_id].size() ||
                    member_offsets[type_id][member] == Missing) return std::nullopt;
                const uint32_t stride = member < member_matrix_strides[type_id].size() ?
                    member_matrix_strides[type_id][member] : Missing;
                const auto bytes = size_of(type.members[member], stride, depth + 1);
                if (!bytes) return std::nullopt;
                end = std::max(end, uint64_t(member_offsets[type_id][member]) + *bytes);
            }
            return end;
        }
        return std::nullopt;
    };
    result.stage = stage;
    bool unknown_push_extent = false;
    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> interface_location_span;
    interface_location_span = [&](uint32_t type_id, uint32_t component,
        uint32_t depth) -> uint32_t {
        if (!id_valid(type_id) || depth >= bound) return Missing;
        const Type& type = types[type_id];
        if (type.opcode == 32)
            return interface_location_span(type.element, component, depth + 1);
        if (type.opcode == 23)
        {
            if (!type.count || type.count > 4 || !id_valid(type.element)) return Missing;
            const Type& scalar = types[type.element];
            if ((scalar.opcode != 21 && scalar.opcode != 22) ||
                (scalar.width != 32 && scalar.width != 64)) return Missing;
            const uint32_t component_count = type.count * (scalar.width / 32);
            return (component + component_count + 3) / 4;
        }
        if (type.opcode == 24)
        {
            const uint32_t column_span = interface_location_span(type.element, component, depth + 1);
            if (!type.count || !column_span || column_span == Missing ||
                type.count > 32 / column_span)
                return Missing;
            return type.count * column_span;
        }
        if (type.opcode == 28)
        {
            const uint32_t length = id_valid(type.length_id) ? constants[type.length_id] : Missing;
            const uint32_t element_span = interface_location_span(type.element, component, depth + 1);
            if (!length || length == Missing || !element_span || element_span == Missing ||
                length > 32 / element_span)
                return Missing;
            return length * element_span;
        }
        if (type.opcode == 21 || type.opcode == 22)
        {
            if (type.width != 32 && type.width != 64) return Missing;
            return (component + type.width / 32 + 3) / 4;
        }
        if (type.opcode == 30)
        {
            uint32_t span = 0;
            bool first_user_member = true;
            for (size_t member = 0; member < type.members.size(); ++member)
            {
                if (member < member_builtins[type_id].size() &&
                    member_builtins[type_id][member]) continue;
                const uint32_t member_component = member < member_components[type_id].size() &&
                    member_components[type_id][member] != Missing ?
                    member_components[type_id][member] :
                    (first_user_member ? component : 0);
                const uint32_t member_span = interface_location_span(
                    type.members[member], member_component, depth + 1);
                if (member_span == Missing || span > UINT32_MAX - member_span)
                    return Missing;
                span += member_span;
                first_user_member = false;
            }
            return span;
        }
        return Missing;
    };
    std::function<void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, bool)> emit_interface;
    emit_interface = [&](uint32_t type_id, uint32_t location, uint32_t component,
        uint32_t storage, uint32_t depth, bool has_location, bool use_member_locations) {
        if (!id_valid(type_id) || depth >= bound)
        {
            result.interface_complete = false;
            return;
        }
        const Type& type = types[type_id];
        if (type.opcode == 28 || type.opcode == 24)
        {
            const uint32_t count = type.opcode == 24 ? type.count :
                (id_valid(type.length_id) ? constants[type.length_id] : Missing);
            const uint32_t element_span = interface_location_span(type.element, component, depth + 1);
            if (!has_location || !count || count == Missing || !element_span ||
                element_span == Missing ||
                count > 32 / element_span || location > UINT32_MAX - count * element_span)
            {
                result.interface_complete = false;
                return;
            }
            for (uint32_t index = 0; index < count; ++index)
                emit_interface(type.element, location + index * element_span,
                    component, storage, depth + 1, true, false);
            return;
        }
        if (type.opcode == 30)
        {
            uint32_t cursor = location;
            bool first_user_member = true;
            if (!has_location)
            {
                if (!use_member_locations)
                {
                    result.interface_complete = false;
                    return;
                }
                cursor = Missing;
                for (size_t member = 0; member < type.members.size(); ++member)
                {
                    if (member < member_builtins[type_id].size() &&
                        member_builtins[type_id][member]) continue;
                    if (member >= member_locations[type_id].size() ||
                        member_locations[type_id][member] == Missing)
                    {
                        result.interface_complete = false;
                        continue;
                    }
                    cursor = std::min(cursor, member_locations[type_id][member]);
                }
                if (cursor == Missing)
                    return;
            }
            for (size_t member = 0; member < type.members.size(); ++member)
            {
                if (member < member_builtins[type_id].size() &&
                    member_builtins[type_id][member]) continue;
                const bool has_member_location = member < member_locations[type_id].size() &&
                    member_locations[type_id][member] != Missing;
                if (use_member_locations && !has_member_location)
                {
                    result.interface_complete = false;
                    continue;
                }
                if (has_location && has_member_location)
                {
                    result.interface_complete = false;
                    continue;
                }
                const uint32_t member_location = use_member_locations ?
                    member_locations[type_id][member] : cursor;
                const uint32_t member_component = member < member_components[type_id].size() &&
                    member_components[type_id][member] != Missing ?
                    member_components[type_id][member] :
                    (first_user_member ? component : 0);
                const uint32_t member_span = interface_location_span(
                    type.members[member], member_component, depth + 1);
                if (member_span == Missing || member_location > UINT32_MAX - member_span)
                {
                    result.interface_complete = false;
                    continue;
                }
                emit_interface(type.members[member], member_location, member_component,
                    storage, depth + 1, true, false);
                if (!use_member_locations)
                    cursor = member_location + member_span;
                first_user_member = false;
            }
            return;
        }

        if (!has_location)
        {
            result.interface_complete = false;
            return;
        }
        ShaderInterfaceVariable io;
        io.location = location;
        io.component = component;
        io.vector_size = type.opcode == 23 ? type.count : 1;
        const uint32_t scalar_id = type.opcode == 23 ? type.element : type_id;
        if (id_valid(scalar_id))
        {
            const Type& scalar = types[scalar_id];
            io.scalar_opcode = scalar.opcode;
            io.width = scalar.width;
            io.signedness = scalar.signedness;
            io.type_known = (scalar.opcode == 21 || scalar.opcode == 22) &&
                (io.width == 32 || io.width == 64) &&
                io.vector_size > 0 && io.vector_size <= 4 && component < 4;
            if (io.type_known)
            {
                const uint32_t component_count = io.vector_size * (io.width / 32);
                io.type_known = io.width == 32 ? component + component_count <= 4 :
                    component + component_count <= 8;
            }
        }
        if (!io.type_known)
        {
            result.interface_complete = false;
            return;
        }
        auto& values = storage == 1 ? result.inputs : result.outputs;
        values.push_back(io);
    };
    const auto append_interface = [&](uint32_t type_id, uint32_t location,
        uint32_t component, uint32_t storage, bool has_location,
        bool use_member_locations) {
        emit_interface(type_id, location, component, storage, 0,
            has_location, use_member_locations);
    };
    for (uint32_t id = 1; id < bound; ++id)
    {
        const Variable& variable = variables[id];
        if (!variable.present) continue;
        if ((variable.storage == 1 || variable.storage == 3) && entry_interfaces[id])
        {
            if (builtins[id]) continue;
            if (locations[id] == Missing)
            {
                uint32_t type_id = variable.type;
                if (id_valid(type_id) && types[type_id].opcode == 32)
                    type_id = types[type_id].element;
                if (id_valid(type_id) && types[type_id].opcode == 28)
                    type_id = types[type_id].element;
                if (!id_valid(type_id) || types[type_id].opcode != 30)
                {
                    result.interface_complete = false;
                    continue;
                }
                append_interface(type_id, 0, components[id], variable.storage, false, true);
                continue;
            }
            uint32_t type_id = variable.type;
            if (id_valid(type_id) && types[type_id].opcode == 32)
                type_id = types[type_id].element;
            if (id_valid(type_id) && types[type_id].opcode == 28 &&
                (stage == 1 || stage == 2 || stage == 3))
                type_id = types[type_id].element; // per-vertex interface array
            append_interface(type_id, locations[id], components[id], variable.storage,
                true, false);
            continue;
        }
        if (variable.storage == 9) // PushConstant
        {
            result.push_constants = true;
            const auto bytes = size_of(variable.type, Missing, 0);
            if (!bytes || *bytes > UINT32_MAX)
                unknown_push_extent = true;
            else
            {
                result.push_constant_extent_known = true;
                result.push_constant_bytes = std::max(result.push_constant_bytes,
                    static_cast<uint32_t>(*bytes));
            }
            continue;
        }
        if (variable.storage != 0 && variable.storage != 2 && variable.storage != 12)
            continue;
        // A declared but unused resource may have no descriptor decorations.
        if (sets[id] == Missing && bindings[id] == Missing)
            continue;
        if (sets[id] == Missing || bindings[id] == Missing)
        {
            error = "SPIR-V reflection: descriptor is missing set or binding";
            result = {};
            return false;
        }
        uint32_t type_id = variable.type;
        uint32_t descriptor_count = 1;
        ShaderDescriptor descriptor;
        descriptor.set = sets[id];
        descriptor.binding = bindings[id];
        for (uint32_t depth = 0; depth < bound; ++depth)
        {
            if (!id_valid(type_id)) break;
            const Type& type = types[type_id];
            if (type.opcode != 32 && type.opcode != 28 && type.opcode != 29) break;
            if (type.opcode == 29)
            {
                descriptor_count = 0;
                descriptor.runtime_sized = true;
            }
            if (type.opcode == 28)
            {
                descriptor.array_length_ids.push_back(type.length_id);
                const uint32_t length = id_valid(type.length_id) ? constants[type.length_id] : Missing;
                if (!length || length == Missing || descriptor_count > UINT32_MAX / length)
                    descriptor_count = 0;
                else descriptor_count *= length;
            }
            type_id = type.element;
        }
        ShaderResourceKind kind = ShaderResourceKind::Unknown;
        if (variable.storage == 12) kind = ShaderResourceKind::StorageBuffer;
        else if (variable.storage == 2)
            kind = id_valid(type_id) && types[type_id].buffer_block ?
                ShaderResourceKind::StorageBuffer : ShaderResourceKind::UniformBuffer;
        else if (id_valid(type_id))
        {
            const Type& type = types[type_id];
            if (type.opcode == 26) kind = ShaderResourceKind::Sampler;
            if (type.opcode == 25) kind = type.sampled == 2 ?
                ShaderResourceKind::StorageImage : ShaderResourceKind::SampledImage;
            if (type.opcode == 27) kind = ShaderResourceKind::CombinedImageSampler;
            if (type.opcode == 5341) kind = ShaderResourceKind::AccelerationStructure;
        }
        descriptor.kind = kind;
        descriptor.count = descriptor_count;
        uint32_t image_type_id = type_id;
        if (id_valid(image_type_id) && types[image_type_id].opcode == 27)
            image_type_id = types[image_type_id].element;
        if (id_valid(image_type_id) && types[image_type_id].opcode == 25)
        {
            const Type& image_type = types[image_type_id];
            switch (image_type.image_dimension)
            {
            case 0: descriptor.image.dimension = ShaderImageDimension::OneD; break;
            case 1: descriptor.image.dimension = ShaderImageDimension::TwoD; break;
            case 2: descriptor.image.dimension = ShaderImageDimension::ThreeD; break;
            case 3: descriptor.image.dimension = ShaderImageDimension::Cube; break;
            case 4: descriptor.image.dimension = ShaderImageDimension::Rect; break;
            case 5: descriptor.image.dimension = ShaderImageDimension::Buffer; break;
            case 6: descriptor.image.dimension = ShaderImageDimension::Subpass; break;
            default: descriptor.image.dimension = ShaderImageDimension::Unknown; break;
            }
            descriptor.image.arrayed = image_type.image_arrayed != 0;
            descriptor.image.multisampled = image_type.image_multisampled != 0;
        }
        result.descriptors.push_back(std::move(descriptor));
    }
    std::sort(result.descriptors.begin(), result.descriptors.end(),
        [](const ShaderDescriptor& a, const ShaderDescriptor& b) {
            return a.set < b.set || (a.set == b.set && a.binding < b.binding);
        });
    std::vector<uint32_t> specialization_counts;
    std::string specialization_error;
    if (resolve_shader_descriptor_counts(result, {}, specialization_counts, specialization_error))
        for (size_t i = 0; i < result.descriptors.size(); ++i)
            result.descriptors[i].count = specialization_counts[i];
    if (unknown_push_extent)
    {
        result.push_constant_extent_known = false;
        result.push_constant_bytes = 0;
    }
    const auto sort_io = [](std::vector<ShaderInterfaceVariable>& values) {
        std::sort(values.begin(), values.end(),
            [](const ShaderInterfaceVariable& a, const ShaderInterfaceVariable& b) {
                return a.location < b.location ||
                    (a.location == b.location && a.component < b.component);
            });
    };
    sort_io(result.inputs);
    sort_io(result.outputs);
    const auto duplicates = [](const std::vector<ShaderInterfaceVariable>& values) {
        for (size_t left = 0; left < values.size(); ++left)
        {
            const ShaderInterfaceVariable& a = values[left];
            const uint64_t a_begin = uint64_t(a.location) * 4 + a.component;
            const uint64_t a_end = a_begin + uint64_t(a.vector_size) * (a.width / 32);
            for (size_t right = left + 1; right < values.size(); ++right)
            {
                const ShaderInterfaceVariable& b = values[right];
                const uint64_t b_begin = uint64_t(b.location) * 4 + b.component;
                const uint64_t b_end = b_begin + uint64_t(b.vector_size) * (b.width / 32);
                if (a_begin < b_end && b_begin < a_end)
                    return true;
            }
        }
        return false;
    };
    if (duplicates(result.inputs) || duplicates(result.outputs))
    {
        error = "SPIR-V reflection: duplicate interface location/component";
        result = {};
        return false;
    }
    error.clear();
    return true;
}

bool validate_shader_stage_link(const ShaderReflection& producer,
    const ShaderReflection& consumer, std::string& error)
{
    if (!producer.interface_complete || !consumer.interface_complete)
    {
        error = "shader stage interface contains an unsupported or incomplete type/layout";
        return false;
    }
    for (const ShaderInterfaceVariable& input : consumer.inputs)
    {
        const auto output = std::find_if(producer.outputs.begin(), producer.outputs.end(),
            [&](const ShaderInterfaceVariable& candidate) {
                return candidate.location == input.location && candidate.component == input.component;
            });
        if (output == producer.outputs.end())
        {
            error = "shader stage input location " + std::to_string(input.location) +
                " component " + std::to_string(input.component) + " has no producer";
            return false;
        }
        if (input.type_known && output->type_known &&
            (input.scalar_opcode != output->scalar_opcode || input.width != output->width ||
                input.signedness != output->signedness ||
                input.vector_size != output->vector_size))
        {
            error = "shader stage interface type mismatch at location " +
                std::to_string(input.location);
            return false;
        }
    }
    error.clear();
    return true;
}
}
