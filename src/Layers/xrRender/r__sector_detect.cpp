#include "stdafx.h"

namespace xray::render::RENDER_NAMESPACE
{
IRender_Sector::sector_id_t R_dsgraph_structure::detect_sector(const Fvector& P)
{
    Fvector dir{ 0, -1, 0 };
    auto sector = detect_sector(P, dir);
    if (sector == IRender_Sector::INVALID_SECTOR_ID)
    {
        dir = { 0, 1, 0 };
        sector = detect_sector(P, dir);
    }
    return sector;
}

IRender_Sector::sector_id_t R_dsgraph_structure::detect_sector(const Fvector& P, Fvector& dir)
{
    // Portals model
    int id1 = -1;
    float range1 = 500.f;
    if (RImplementation.rmPortals)
    {
        Sectors_xrc.ray_query(CDB::OPT_ONLYNEAREST, RImplementation.rmPortals, P, dir, range1);
        if (Sectors_xrc.r_count())
        {
            CDB::RESULT* RP1 = Sectors_xrc.r_begin();
            id1 = RP1->id;
            range1 = RP1->range;
        }
    }

    // Geometry model
    int id2 = -1;
    float range2 = range1;
    Sectors_xrc.ray_query(CDB::OPT_ONLYNEAREST, g_pGameLevel->ObjectSpace.GetStaticModel(), P, dir, range2);
    if (Sectors_xrc.r_count())
    {
        CDB::RESULT* RP2 = Sectors_xrc.r_begin();
        id2 = RP2->id;
        range2 = RP2->range;
    }

    // Select ID
    int ID;
    bool portalHit = false;
    if (id1 >= 0)
    {
        if (id2 >= 0)
        {
            portalHit = range1 <= range2 + EPS;
            ID = portalHit ? id1 : id2; // both were found
        }
        else
        {
            portalHit = true;
            ID = id1; // only id1 found
        }
    }
    else if (id2 >= 0)
        ID = id2; // only id2 found
    else
        return IRender_Sector::INVALID_SECTOR_ID;

    if (portalHit)
    {
        // Take sector, facing to our point from portal
        CDB::TRI* pTri = RImplementation.rmPortals->get_tris() + ID;
        CPortal* pPortal = Portals[pTri->dummy];
        return pPortal->getSectorFacing(P)->unique_id;
    }
    // Take triangle at ID and use it's Sector
    CDB::TRI* pTri = g_pGameLevel->ObjectSpace.GetStaticTris() + ID;
    return static_cast<IRender_Sector::sector_id_t>(pTri->sector);
}
#if defined(XR_PLATFORM_ANDROID)
void R_dsgraph_structure::audit_camera_sector(const Fvector& position, IRender_Sector::sector_id_t sector)
{
    auto& audit = sector_audit;
    if (audit.query == 4)
    {
        if (audit.samples >= 8 || (audit.samples && Device.dwTimeContinual - audit.last_sample < 5000))
            return;
        const u32 samples = audit.samples + 1;
        audit = {};
        audit.position = position;
        audit.query = 0;
        audit.frame = Device.dwFrame;
        audit.last_sample = Device.dwTimeContinual;
        audit.samples = samples;
        audit.sector = sector;
        for (u32 query = 0; query < 4; ++query)
        {
            const auto* model = query % 2 ? RImplementation.rmPortals : g_pGameLevel->ObjectSpace.GetStaticModel();
            const Fvector direction{0, query < 2 ? -1.f : 1.f, 0};
            auto& ref = audit.reference[query];
            ref.origin = {position.x, position.y, position.z};
            ref.direction = {0, direction.y, 0};
            if (!model)
                continue;
            Sectors_xrc.ray_query(CDB::OPT_ONLYNEAREST, model, position, direction, 500.f);
            if (Sectors_xrc.r_count())
            {
                audit.accelerated_id[query] = Sectors_xrc.r_begin()->id;
                audit.accelerated_range[query] = Sectors_xrc.r_begin()->range;
            }
        }
        Msg("[sector-audit] begin frame=%u pos=(%.5f,%.5f,%.5f) selected=%u tree-path=%s triangles=%u",
            audit.frame, position.x, position.y, position.z, static_cast<u32>(sector),
            CPU::HasSSE ? "simd" : "scalar", g_pGameLevel->ObjectSpace.GetStaticModel()->get_tris_count());
    }

    const u64 start = CPU::QPC();
    const u64 budget = _max(u64(1), CPU::qpc_freq / 4000); // 0.25 ms per frame.
    for (u32 tested = 0; audit.query < 4 && tested < 2048;)
    {
        const u32 query = audit.query;
        const auto* model = query % 2 ? RImplementation.rmPortals : g_pGameLevel->ObjectSpace.GetStaticModel();
        auto& ref = audit.reference[query];
        if (model && audit.next_triangle < model->get_tris_count())
        {
            const u32 id = audit.next_triangle++;
            const auto& tri = model->get_tris()[id];
            const auto* verts = model->get_verts();
            const auto& a = verts[tri.verts[0]];
            const auto& b = verts[tri.verts[1]];
            const auto& c = verts[tri.verts[2]];
            ref.test(id, {a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z});
            ++tested;
            if (tested % 64 == 0 && CPU::QPC() - start >= budget)
                break;
            continue;
        }

        const int accelerated = audit.accelerated_id[query];
        const auto sectorFor = [&](int id) -> u32
        {
            if (id < 0 || !model)
                return IRender_Sector::INVALID_SECTOR_ID;
            const auto& tri = model->get_tris()[id];
            if (query % 2)
                return tri.dummy < Portals.size() ? Portals[tri.dummy]->getSectorFacing(audit.position)->unique_id :
                    IRender_Sector::INVALID_SECTOR_ID;
            return tri.sector;
        };
        const bool sameHit = (accelerated < 0 && ref.nearest < 0) ||
            (accelerated >= 0 && ref.nearest >= 0 && sectorFor(accelerated) == sectorFor(ref.nearest) &&
                _abs(double(audit.accelerated_range[query]) - ref.distance) < 0.001);
        Msg("[sector-audit] frame=%u query=%s/%s tree=%d sector=%u range=%.6f "
            "reference=%d sector=%u range=%.6f match=%d",
            audit.frame, query < 2 ? "down" : "up", query % 2 ? "portals" : "static",
            accelerated, sectorFor(accelerated), audit.accelerated_range[query],
            ref.nearest, sectorFor(ref.nearest), ref.distance, sameHit);
        if (!sameHit && model)
        {
            for (int id : {accelerated, ref.nearest})
            {
                if (id < 0)
                    continue;
                const auto& tri = model->get_tris()[id];
                const auto* verts = model->get_verts();
                const auto& a = verts[tri.verts[0]];
                const auto& b = verts[tri.verts[1]];
                const auto& c = verts[tri.verts[2]];
                Msg("[sector-audit] triangle=%d metadata=%08x a=(%.6f,%.6f,%.6f) "
                    "b=(%.6f,%.6f,%.6f) c=(%.6f,%.6f,%.6f)", id, tri.dummy,
                    a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
            }
        }
        ++audit.query;
        audit.next_triangle = 0;
    }
    audit.work_ms += 1000.0 * double(CPU::QPC() - start) / double(CPU::qpc_freq);
    if (audit.query == 4)
    {
        const auto resolveSector = [&](bool reference) -> u32
        {
            for (u32 base : {0u, 2u})
            {
                const auto id = [&](u32 q) { return reference ? audit.reference[q].nearest : audit.accelerated_id[q]; };
                const auto distance = [&](u32 q) { return reference ? audit.reference[q].distance : double(audit.accelerated_range[q]); };
                if (id(base) < 0 && id(base + 1) < 0)
                    continue;
                const bool portal = id(base + 1) >= 0 &&
                    (id(base) < 0 || distance(base + 1) <= distance(base) + EPS);
                if (!portal)
                    return g_pGameLevel->ObjectSpace.GetStaticTris()[id(base)].sector;
                const u32 portalId = RImplementation.rmPortals->get_tris()[id(base + 1)].dummy;
                return portalId < Portals.size() ? Portals[portalId]->getSectorFacing(audit.position)->unique_id :
                    IRender_Sector::INVALID_SECTOR_ID;
            }
            return IRender_Sector::INVALID_SECTOR_ID;
        };
        Msg("[sector-audit] end frame=%u finished=%u selected=%u tree-sector=%u reference-sector=%u work=%.3fms",
            audit.frame, Device.dwFrame, static_cast<u32>(audit.sector), resolveSector(false), resolveSector(true), audit.work_ms);
        audit.last_sample = Device.dwTimeContinual;
    }
}
#endif
} // namespace xray::render::RENDER_NAMESPACE
