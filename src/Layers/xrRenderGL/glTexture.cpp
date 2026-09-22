// Texture.cpp: implementation of the CTexture class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include <gli/gli.hpp>

namespace xray::render::RENDER_NAMESPACE
{
void fix_texture_name(pstr fn)
{
    pstr _ext = strext(fn);
    if (_ext &&
        (0 == xr_stricmp(_ext, ".tga") ||
            0 == xr_stricmp(_ext, ".dds") ||
            0 == xr_stricmp(_ext, ".bmp") ||
            0 == xr_stricmp(_ext, ".ogm")))
        *_ext = 0;
}

int get_texture_load_lod(LPCSTR fn)
{
    CInifile::Sect& sect = pSettings->r_section("reduce_lod_texture_list");

    for (const auto& item : sect.Data)
    {
        if (strstr(fn, item.first.c_str()))
        {
            if (psTextureLOD < 1)
                return 0;
            if (psTextureLOD < 3)
                return 1;
            return 2;
        }
    }

    if (psTextureLOD < 2)
        return 0;
    if (psTextureLOD < 4)
        return 1;
    return 2;
}

u32 calc_texture_size(int lod, u32 mip_cnt, size_t orig_size)
{
    if (1 == mip_cnt)
        return orig_size;

    int _lod = lod;
    float res = float(orig_size);

    while (_lod > 0)
    {
        --_lod;
        res -= res / 1.333f;
    }
    return iFloor(res);
}

#if defined(XR_PLATFORM_ANDROID)
void clear_texture_gl_errors()
{
    while (glGetError() != GL_NO_ERROR)
    {
    }
}

bool check_texture_gl_error(cpcstr operation, cpcstr filename)
{
    bool valid = true;
    for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError())
    {
        Msg("! OpenGL ES: 0x%x during %s for texture '%s'", error, operation, filename);
        valid = false;
    }
    return valid;
}

bool decode_compressed_texture(gli::texture& texture)
{
    if (!gli::is_compressed(texture.format()))
        return true;

    if (!gli::has_decoder(texture.format()))
        return false;

    constexpr gli::format decoded_format = gli::FORMAT_RGBA8_UNORM_PACK8;
    switch (texture.target())
    {
    case gli::TARGET_1D:
        texture = gli::convert(gli::texture1d(texture), decoded_format);
        return true;
    case gli::TARGET_1D_ARRAY:
        texture = gli::convert(gli::texture1d_array(texture), decoded_format);
        return true;
    case gli::TARGET_2D:
        texture = gli::convert(gli::texture2d(texture), decoded_format);
        return true;
    case gli::TARGET_2D_ARRAY:
        texture = gli::convert(gli::texture2d_array(texture), decoded_format);
        return true;
    case gli::TARGET_3D:
        texture = gli::convert(gli::texture3d(texture), decoded_format);
        return true;
    case gli::TARGET_CUBE:
        texture = gli::convert(gli::texture_cube(texture), decoded_format);
        return true;
    case gli::TARGET_CUBE_ARRAY:
        texture = gli::convert(gli::texture_cube_array(texture), decoded_format);
        return true;
    default:
        return false;
    }
}

bool supports_compressed_texture_format(GLenum internalFormat)
{
    GLint count = 0;
    glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &count);
    if (count <= 0)
        return false;

    xr_vector<GLint> formats(static_cast<size_t>(count));
    glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, formats.data());
    return std::find(formats.begin(), formats.end(), static_cast<GLint>(internalFormat)) != formats.end();
}
#endif

GLuint CRender::texture_load(LPCSTR fRName, u32& ret_msize, GLenum& ret_desc,
    GLint* ret_width, GLint* ret_height)
{
    ret_msize = 0;
    R_ASSERT1_CURE(fRName && fRName[0], { return 0; });

    GLuint pTexture = 0;
    string_path fn;
    {
        // make file name
        string_path fname;
        xr_strcpy(fname, fRName);
        fix_texture_name(fname);

        // Call to FS.exist WRITES to fn !

        if (!FS.exist(fn, "$game_textures$", fname, ".dds") && strstr(fname, "_bump"))
        {
            Msg("! Fallback to default bump map: %s", fname);
            if (strstr(fname, "_bump#"))
                R_ASSERT1_CURE(FS.exist(fn, "$game_textures$", "ed\\ed_dummy_bump#", ".dds"), return 0);
            else
                R_ASSERT1_CURE(FS.exist(fn, "$game_textures$", "ed\\ed_dummy_bump", ".dds"), return 0);
        }
        else
        {
            bool exist = false;

            for (cpcstr folder : { "$level$", "$game_saves$", "$game_textures$" })
            {
                exist = FS.exist(fn, folder, fname, ".dds");
                if (exist)
                    break;
            }

            if (!exist)
            {
                Msg("! Can't find texture '%s'", fname);
                if (!FS.exist(fn, "$game_textures$", "ed\\ed_not_existing_texture", ".dds"))
                    return 0;
            }
        }
    }

    // Load and get header
    IReader* S = FS.r_open(fn);
    R_ASSERT2_CURE(S, fn, { return 0; });
    size_t img_size = S->length();
#if defined(XR_PLATFORM_ANDROID)
    CTimer androidTextureTimer;
    androidTextureTimer.Start();
#endif
#ifdef DEBUG
    Msg("* Loaded: %s[%d]b", fn, img_size);
#endif // DEBUG
    gli::texture texture = gli::load((char*)S->pointer(), img_size);
    R_ASSERT2(!texture.empty(), fn);

#if defined(XR_PLATFORM_ANDROID)
    // GLES 3.0 does not require desktop S3TC/BC texture formats.  CoP ships
    // many DDS files in DXT form.  Preserve hardware-supported compressed
    // formats (Adreno normally exposes S3TC) to avoid multiplying level
    // texture memory by four; decode only formats the current device cannot
    // upload.  The previous decode-all path could exhaust a 32-bit process
    // near the end of a large CoP level load.
    if (gli::is_compressed(texture.format()))
    {
        const gli::gl compressedGL(gli::gl::PROFILE_ES30);
        const auto compressedFormat = compressedGL.translate(texture.format(), texture.swizzles());
        if (!supports_compressed_texture_format(compressedFormat.Internal))
        {
            Msg("[texture-trace] decode-begin name='%s' source=%zu bytes format=0x%x",
                fRName, img_size, compressedFormat.Internal);
            CTimer decodeTimer;
            decodeTimer.Start();
            if (!decode_compressed_texture(texture))
            {
                Msg("! Android GLES: no decoder for compressed texture '%s'", fn);
                FS.r_close(S);
                return 0;
            }
            Msg("[texture-trace] decode-end name='%s' elapsed=%llu ms",
                fRName, static_cast<unsigned long long>(decodeTimer.GetElapsed_ms()));
        }
        // DeferredUpload reports bounded progress.  Logging every decoded DDS
        // made a large CoP level produce thousands of lines and also caused
        // the launcher diagnostics view to stutter badly.
    }
#endif

    const u32 mip_cnt = static_cast<u32>(texture.levels());

#if defined(XR_PLATFORM_ANDROID)
    gli::gl GL(gli::gl::PROFILE_ES30);
#else
    gli::gl GL(gli::gl::PROFILE_GL33);
#endif

    gli::gl::format const format = GL.translate(texture.format(), texture.swizzles());
    GLenum target = GL.translate(texture.target());

#if defined(XR_PLATFORM_ANDROID)
    // Attribute an error to the operation that actually caused it.  Release
    // builds make CHK_GL a pass-through, so stale driver errors otherwise get
    // reported later as a bogus glTexStorage2D failure.
    clear_texture_gl_errors();
#endif
    glGenTextures(1, &pTexture);
    glBindTexture(target, pTexture);

    glTexParameteri(target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(texture.levels() - 1));

    if (gli::gl::EXTERNAL_RED != format.External) // skip for proper greyscale-alpha font textures
#if defined(XR_PLATFORM_ANDROID)
    {
        // GL_TEXTURE_SWIZZLE_RGBA is a desktop aggregate pname.  GLES 3.x
        // exposes only the four component pnames and returns GL_INVALID_ENUM
        // for the aggregate form used by the desktop renderer.
        static constexpr GLenum SwizzleNames[] =
        {
            GL_TEXTURE_SWIZZLE_R,
            GL_TEXTURE_SWIZZLE_G,
            GL_TEXTURE_SWIZZLE_B,
            GL_TEXTURE_SWIZZLE_A
        };
        for (size_t component = 0; component < std::size(SwizzleNames); ++component)
            glTexParameteri(target, SwizzleNames[component], format.Swizzles[component]);
    }
#else
        glTexParameteriv(target, GL_TEXTURE_SWIZZLE_RGBA, &format.Swizzles[gli::SWIZZLE_RED]);
#endif

#if defined(XR_PLATFORM_ANDROID)
    if (!pTexture || !check_texture_gl_error("texture creation and parameters", fn))
    {
        if (pTexture)
            glDeleteTextures(1, &pTexture);
        FS.r_close(S);
        return 0;
    }
#endif

    glm::tvec3<GLsizei> const tex_extent(texture.extent());
    if (ret_width)
        *ret_width = tex_extent.x;
    if (ret_height)
        *ret_height = tex_extent.y;

#if !defined(XR_PLATFORM_ANDROID)
    GLenum err;
#endif
    switch (texture.target())
    {
    case gli::TARGET_2D:
    case gli::TARGET_CUBE:
        glTexStorage2D(target, static_cast<GLint>(texture.levels()), format.Internal,
                       tex_extent.x, tex_extent.y);
#if defined(XR_PLATFORM_ANDROID)
        if (!check_texture_gl_error("2D immutable storage allocation", fn))
        {
            glDeleteTextures(1, &pTexture);
            FS.r_close(S);
            return 0;
        }
#else
        err = glGetError();
        if (err != GL_NO_ERROR)
        {
            VERIFY(err == GL_NO_ERROR);
            Msg("! OpenGL: 0x%x: Invalid 2D texture: '%s'", err, fn);
        }
#endif
        break;
    case gli::TARGET_3D:
    case gli::TARGET_CUBE_ARRAY:
        glTexStorage3D(target, static_cast<GLint>(texture.levels()), format.Internal,
                       tex_extent.x, tex_extent.y, tex_extent.z);
#if defined(XR_PLATFORM_ANDROID)
        if (!check_texture_gl_error("3D immutable storage allocation", fn))
        {
            glDeleteTextures(1, &pTexture);
            FS.r_close(S);
            return 0;
        }
#else
        err = glGetError();
        if (err != GL_NO_ERROR)
        {
            VERIFY(err == GL_NO_ERROR);
            Msg("! OpenGL: 0x%x: Invalid 3D texture: '%s'", err, fn);
        }
#endif
        break;
    default:
        NODEFAULT;
        break;
    }

    for (size_t layer = 0; layer < texture.layers(); ++layer)
    {
        for (size_t face = 0; face < texture.faces(); ++face)
        {
            for (size_t level = 0; level < texture.levels(); ++level)
            {
                glm::tvec3<GLsizei> const tex_level_extent(texture.extent(level));
                GLenum sub_target = gli::is_target_cube(texture.target())
                         ? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face)
                         : target;

                switch (texture.target())
                {
                case gli::TARGET_2D:
                case gli::TARGET_CUBE:
                {
                    if (gli::is_compressed(texture.format()))
                    {
                        glCompressedTexSubImage2D(sub_target, static_cast<GLint>(level),
                                    0, 0, tex_level_extent.x, tex_level_extent.y,
                                    format.Internal, static_cast<GLsizei>(texture.size(level)),
                                    texture.data(layer, face, level));
#if defined(XR_PLATFORM_ANDROID)
                        if (!check_texture_gl_error("compressed 2D texture upload", fn))
                        {
                            glDeleteTextures(1, &pTexture);
                            FS.r_close(S);
                            return 0;
                        }
#else
                        err = glGetError();
                        if (err != GL_NO_ERROR)
                        {
                            VERIFY(err == GL_NO_ERROR);
                            Msg("! OpenGL: 0x%x: Invalid 2D compressed subtexture: '%s'", err, fn);
                        }
#endif
                    }
                    else
                    {
                        glTexSubImage2D(sub_target, static_cast<GLint>(level),
                                    0, 0, tex_level_extent.x, tex_level_extent.y,
                                    format.External, format.Type,
                                    texture.data(layer, face, level));
#if defined(XR_PLATFORM_ANDROID)
                        if (!check_texture_gl_error("2D texture upload", fn))
                        {
                            glDeleteTextures(1, &pTexture);
                            FS.r_close(S);
                            return 0;
                        }
#else
                        err = glGetError();
                        if (err != GL_NO_ERROR)
                        {
                            VERIFY(err == GL_NO_ERROR);
                            Msg("! OpenGL: 0x%x: Invalid 2D subtexture: '%s'", err, fn);
                        }
#endif

                    }
                    break;
                }
                case gli::TARGET_3D:
                case gli::TARGET_CUBE_ARRAY:
                {
                    if (gli::is_compressed(texture.format()))
                    {
                        glCompressedTexSubImage3D(target, static_cast<GLint>(level),
                                    0, 0, 0, tex_level_extent.x, tex_level_extent.y, tex_level_extent.z,
                                    format.Internal, static_cast<GLsizei>(texture.size(level)),
                                    texture.data(layer, face, level));
#if defined(XR_PLATFORM_ANDROID)
                        if (!check_texture_gl_error("compressed 3D texture upload", fn))
                        {
                            glDeleteTextures(1, &pTexture);
                            FS.r_close(S);
                            return 0;
                        }
#else
                        err = glGetError();
                        if (err != GL_NO_ERROR)
                        {
                            VERIFY(err == GL_NO_ERROR);
                            Msg("! OpenGL: 0x%x: Invalid compressed 3D subtexture: '%s'", err, fn);
                        }
#endif
                    }
                    else
                    {
                        glTexSubImage3D(target, static_cast<GLint>(level),
                                    0, 0, 0, tex_level_extent.x, tex_level_extent.y, tex_level_extent.z,
                                    format.External, format.Type,
                                    texture.data(layer, face, level));
#if defined(XR_PLATFORM_ANDROID)
                        if (!check_texture_gl_error("3D texture upload", fn))
                        {
                            glDeleteTextures(1, &pTexture);
                            FS.r_close(S);
                            return 0;
                        }
#else
                        err = glGetError();
                        if (err != GL_NO_ERROR)
                        {
                            VERIFY(err == GL_NO_ERROR);
                            Msg("! OpenGL: 0x%x: Invalid 3D subtexture: '%s'", err, fn);
                        }
#endif
                    }
                    break;
                }
                default:
                    NODEFAULT;
                    break;
                }
            }
        }
    }

    FS.r_close(S);

    xr_strlwr(fn);
    ret_desc = target;
    int img_loaded_lod = is_target_cube(texture.target()) ? 0 : get_texture_load_lod(fn);
    ret_msize = calc_texture_size(img_loaded_lod, mip_cnt, img_size);
#if defined(XR_PLATFORM_ANDROID)
    const u64 textureElapsed = androidTextureTimer.GetElapsed_ms();
    if (textureElapsed >= 500)
        Msg("[texture-trace] slow-load name='%s' elapsed=%llu ms source=%zu bytes extent=%dx%d levels=%u compressed=%d",
            fRName, static_cast<unsigned long long>(textureElapsed), img_size,
            tex_extent.x, tex_extent.y, mip_cnt, gli::is_compressed(texture.format()) ? 1 : 0);
#endif
    return pTexture;
}
} // namespace xray::render::RENDER_NAMESPACE
