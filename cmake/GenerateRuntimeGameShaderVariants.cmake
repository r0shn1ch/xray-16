include_guard(GLOBAL)

function(_xray_cpp_string_literal input output)
    string(REPLACE "\\" "\\\\" escaped "${input}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    set(${output} "${escaped}" PARENT_SCOPE)
endfunction()

function(xray_generate_runtime_game_shader_variants manifest output_header)
    if (NOT EXISTS "${manifest}")
        message(FATAL_ERROR "Runtime Vulkan shader variant manifest is missing: ${manifest}")
    endif()

    file(READ "${manifest}" manifest_json)
    string(JSON variant_count ERROR_VARIABLE json_error LENGTH "${manifest_json}" variants)
    if (NOT json_error STREQUAL "NOTFOUND" OR variant_count LESS 1)
        message(FATAL_ERROR "Runtime Vulkan shader variant manifest is invalid: ${manifest}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest}")

    get_filename_component(output_directory "${output_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    file(WRITE "${output_header}"
        "#pragma once\n"
        "namespace xray::render::vulkan::generated {\n"
        "struct RuntimeGameShaderVariant { const char* name; const char* source; const char* stage; const char* entry; const char* defines; };\n"
        "inline constexpr RuntimeGameShaderVariant RuntimeGameShaderVariants[] = {\n")

    set(seen_names)
    math(EXPR last_variant "${variant_count} - 1")
    foreach(index RANGE 0 ${last_variant})
        string(JSON source GET "${manifest_json}" variants ${index} source)
        string(JSON stage GET "${manifest_json}" variants ${index} stage)
        string(JSON output GET "${manifest_json}" variants ${index} output)
        string(JSON entry ERROR_VARIABLE entry_error GET "${manifest_json}" variants ${index} entry)
        if (NOT entry_error STREQUAL "NOTFOUND")
            set(entry main)
        endif()

        # Runtime VFS compilation can only use sources packaged under the game
        # shader root. Engine-internal helpers remain precompiled SPIR-V assets.
        if (NOT output MATCHES "^vk/.+\\.spv$" OR IS_ABSOLUTE "${source}" OR
            "${source}" MATCHES "(^|/)\\.\\.(/|$)")
            continue()
        endif()

        string(REGEX REPLACE "\\.spv$" "" shader_name "${output}")
        if (shader_name IN_LIST seen_names)
            message(FATAL_ERROR "Duplicate runtime Vulkan shader variant: ${shader_name}")
        endif()
        list(APPEND seen_names "${shader_name}")

        set(define_list)
        string(JSON define_count ERROR_VARIABLE define_error LENGTH "${manifest_json}" variants ${index} define)
        if (define_error STREQUAL "NOTFOUND")
            if (define_count GREATER 0)
                math(EXPR last_define "${define_count} - 1")
                foreach(define_index RANGE 0 ${last_define})
                    string(JSON define GET "${manifest_json}" variants ${index} define ${define_index})
                    if (define MATCHES "\\|")
                        message(FATAL_ERROR "Runtime Vulkan shader define contains reserved separator '|': ${define}")
                    endif()
                    if (define_list)
                        string(APPEND define_list "|")
                    endif()
                    string(APPEND define_list "${define}")
                endforeach()
            endif()
        endif()

        set(source "vk/${source}")
        _xray_cpp_string_literal("${shader_name}" name_literal)
        _xray_cpp_string_literal("${source}" source_literal)
        _xray_cpp_string_literal("${stage}" stage_literal)
        _xray_cpp_string_literal("${entry}" entry_literal)
        _xray_cpp_string_literal("${define_list}" defines_literal)
        file(APPEND "${output_header}"
            "    {\"${name_literal}\", \"${source_literal}\", \"${stage_literal}\", \"${entry_literal}\", \"${defines_literal}\"},\n")
    endforeach()

    if (NOT seen_names)
        message(FATAL_ERROR "Runtime Vulkan shader variant manifest has no VFS-backed outputs: ${manifest}")
    endif()
    file(APPEND "${output_header}" "};\n} // namespace xray::render::vulkan::generated\n")
endfunction()
