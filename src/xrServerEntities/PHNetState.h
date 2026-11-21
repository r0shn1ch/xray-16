#pragma once

#include "xrCore/_vector3d.h"
#include "xrCore/_quaternion.h"
#include "xrCommon/xr_vector.h"

class NET_Packet;

struct XRPHYSICS_API SPHNetState
{
    Fvector linear_vel;
    Fvector angular_vel;
    Fvector force;
    Fvector torque;
    Fvector position;
    Fvector previous_position;
    union
    {
        Fquaternion quaternion;
        struct
        {
            Fvector accel;
            float max_velocity;
        };
    };
    Fquaternion previous_quaternion;
    bool enabled;
    void net_Export(NET_Packet& P);
    void net_Import(NET_Packet& P);
    void net_Import(IReader& P);
    void net_Save(NET_Packet& P);
    void net_Load(NET_Packet& P);
    void net_Load(IReader& P);
    void net_Save(NET_Packet& P, const Fvector& min, const Fvector& max);
    void net_Load(NET_Packet& P, const Fvector& min, const Fvector& max);
    void net_Load(IReader& P, const Fvector& min, const Fvector& max);

private:
    template <typename src>
    void read(src& P);
    template <typename src>
    void read(src& P, const Fvector& min, const Fvector& max);
};

using PHNETSTATE_VECTOR = xr_vector<SPHNetState>;

struct XRPHYSICS_API SPHBonesData
{
    static constexpr u16 ExtendedVisibilityWordCount = 3; // visibility for bone IDs 64..255
    u64 bones_mask;
    u64 extended_bones_visibility[ExtendedVisibilityWordCount];
    u16 root_bone;
    PHNETSTATE_VECTOR bones;
    Fvector m_min;
    Fvector m_max;

    SPHBonesData();
    void net_Save(NET_Packet& P, bool save_extended_visibility = true);
    void net_Load(NET_Packet& P, bool load_extended_visibility = true);
    void net_Load(IReader& P);
    void set_min_max(const Fvector& _min, const Fvector& _max);
    const Fvector& get_min() const { return m_min; }
    const Fvector& get_max() const { return m_max; }
};
