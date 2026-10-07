"""Optional Blender renderer for the synthetic fire 2D/3D manifest.

Run this file with Blender's bundled Python, not the project virtualenv::

    blender -b --python blender_render_synthetic_fire.py -- \
      --dataset working/synthetic_fire_3d_v3 \
      --split test --max-images 24 \
      --output-dir working/synthetic_fire_3d_v3/blender_test

The script keeps the existing metric contract. It reads the scene/camera/fire
values from ``manifest.jsonl`` and adds rendered RGB, depth and fire-mask paths
to a new ``blender_manifest.jsonl``. A missing Blender installation does not
affect the dependency-light procedural generator or the ray-casting evaluator.

This is a controlled renderer upgrade, not a claim that the result is a real
CCTV domain. It creates a measured-coordinate room, emissive flame geometry and
optional smoke-like transparent blobs; the next research step can replace the
fire material with a volumetric/particle asset without changing the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except ImportError as exc:  # pragma: no cover - executed only outside Blender
    raise SystemExit(
        "Blender Python API (bpy) is required. Run with 'blender -b --python "
        "blender_render_synthetic_fire.py -- ...'."
    ) from exc


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def material_principled(
    name: str,
    base_color: tuple[float, float, float, float],
    roughness: float = 0.65,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
    alpha: float = 1.0,
) -> Any:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    if shader is None:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = base_color
    shader.inputs["Roughness"].default_value = roughness
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = emission or (0.0, 0.0, 0.0, 1.0)
        shader.inputs["Emission Strength"].default_value = emission_strength
    elif "Emission" in shader.inputs:
        shader.inputs["Emission"].default_value = emission or (0.0, 0.0, 0.0, 1.0)
        shader.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
        shader.inputs["Alpha"].default_value = alpha
    return material


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def create_mesh_object(
    name: str,
    vertices: list[list[float]],
    faces: list[list[int]],
    labels: list[str] | None = None,
) -> Any:
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if labels:
        names = sorted(set(labels))
        materials = {
            label: material_principled(
                f"Room_{label}",
                {
                    "floor": (0.18, 0.14, 0.09, 1.0),
                    "back_wall": (0.20, 0.22, 0.26, 1.0),
                    "left_wall": (0.16, 0.18, 0.22, 1.0),
                    "right_wall": (0.16, 0.18, 0.22, 1.0),
                    "ceiling": (0.10, 0.11, 0.14, 1.0),
                    "furniture": (0.22, 0.12, 0.06, 1.0),
                    "obstacle": (0.12, 0.15, 0.18, 1.0),
                }.get(label, (0.20, 0.20, 0.20, 1.0)),
            )
            for label in names
        }
        for label in names:
            obj.data.materials.append(materials[label])
        for index, polygon in enumerate(obj.data.polygons):
            label = labels[index] if index < len(labels) else labels[0]
            polygon.material_index = names.index(label)
    return obj


def configure_camera(row: dict[str, Any], width: int, height: int) -> Any:
    camera_data = bpy.data.cameras.new("SyntheticCamera")
    camera = bpy.data.objects.new("SyntheticCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "PERSP"
    K = row["camera"]["K"]
    fx = float(K[0][0])
    cx = float(K[0][2])
    camera_data.sensor_width = 36.0
    camera_data.lens = max(1.0, fx * camera_data.sensor_width / max(width, 1))
    # Blender's camera local frame is x=right, y=up, -z=forward. The manifest
    # uses x=right, y=down, z=forward.
    R_cv = Matrix(row["camera"]["R_world_to_camera"])
    cv_to_blender = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    R_blender = cv_to_blender @ R_cv
    camera.rotation_euler = R_blender.transposed().to_euler()
    camera.location = Vector(row["camera"]["camera_position"])
    # Approximate principal-point offset. The exact K remains in the manifest.
    camera_data.shift_x = (cx - width * 0.5) / max(width, 1)
    bpy.context.scene.camera = camera
    return camera


def create_fire(point: list[float], sample_id: str, scale: float = 1.0) -> list[Any]:
    """Create a small emissive flame cluster and return its objects."""
    x, y, z = [float(value) for value in point]
    objects: list[Any] = []
    flame_materials = [
        material_principled(
            f"FireOuter_{sample_id}",
            (0.70, 0.025, 0.002, 1.0),
            roughness=0.35,
            emission=(1.0, 0.025, 0.001, 1.0),
            emission_strength=5.0,
        ),
        material_principled(
            f"FireMiddle_{sample_id}",
            (1.0, 0.18, 0.005, 1.0),
            roughness=0.30,
            emission=(1.0, 0.10, 0.002, 1.0),
            emission_strength=8.0,
        ),
        material_principled(
            f"FireCore_{sample_id}",
            (1.0, 0.78, 0.12, 1.0),
            roughness=0.25,
            emission=(1.0, 0.55, 0.06, 1.0),
            emission_strength=12.0,
        ),
    ]
    for index, (radius, depth, material) in enumerate(
        ((0.42, 0.55, flame_materials[0]), (0.29, 0.95, flame_materials[1]), (0.16, 1.28, flame_materials[2]))
    ):
        bpy.ops.mesh.primitive_cone_add(
            vertices=20,
            radius1=radius * scale,
            radius2=0.015 * scale,
            depth=depth * scale,
            location=(x + 0.06 * index * scale, y, z + depth * 0.5 * scale),
        )
        obj = bpy.context.object
        obj.name = f"Fire_{sample_id}_{index}"
        obj.data.materials.append(material)
        obj.pass_index = 1
        objects.append(obj)
    bpy.ops.object.light_add(type="POINT", location=(x, y, z + 0.7 * scale))
    light = bpy.context.object
    light.name = f"FireLight_{sample_id}"
    light.data.energy = 260.0 * scale
    light.data.color = (1.0, 0.14, 0.015)
    objects.append(light)
    return objects


def configure_world_and_render(scene: Any, width: int, height: int, output_dir: Path, sample_id: str) -> None:
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except (TypeError, ValueError):
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_dir / "rgb" / f"{sample_id}.png")
    scene.render.film_transparent = False
    world = scene.world or bpy.data.worlds.new("SyntheticWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.006, 0.008, 0.012, 1.0)
        background.inputs["Strength"].default_value = 0.25

    bpy.ops.object.light_add(type="AREA", location=(0.0, 2.0, 3.5))
    area = bpy.context.object
    area.name = "RoomFillLight"
    area.data.energy = 850.0
    area.data.shape = "RECTANGLE"
    area.data.size = 8.0

    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str(output_dir / "depth")
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.file_slots[0].path = f"{sample_id}_depth_"
    links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    mask_output = nodes.new("CompositorNodeOutputFile")
    mask_output.base_path = str(output_dir / "mask")
    mask_output.format.file_format = "PNG"
    mask_output.file_slots[0].path = f"{sample_id}_mask_"
    id_mask = nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    links.new(render_layers.outputs["IndexOB"], id_mask.inputs[0])
    links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True


def render_rows(args: argparse.Namespace) -> None:
    dataset = args.dataset
    rows = load_jsonl(dataset / "manifest.jsonl")
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if args.max_images > 0:
        rows = rows[: args.max_images]
    mesh = load_json(dataset / "room_mesh.json")
    output_dir = args.output_dir
    for subdir in ("rgb", "depth", "mask"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    for row in rows:
        clear_scene()
        scene = bpy.context.scene
        scene.render.use_file_extension = True
        width, height = [int(value) for value in row["image_size"]]
        create_mesh_object(
            "RoomMesh",
            mesh["vertices"],
            mesh["faces"],
            mesh.get("face_labels"),
        )
        configure_camera(row, width, height)
        fire_object = None
        if int(row.get("fire_visible", row.get("has_fire", 0))) == 1 and row.get("fire_xyz_world"):
            fire_object = create_fire(row["fire_xyz_world"], row["sample_id"])
        configure_world_and_render(scene, width, height, output_dir, row["sample_id"])
        bpy.ops.render.render(write_still=True)
        result = dict(row)
        result["renderer"] = "blender_eevee"
        result["rendered_image_path"] = str((output_dir / "rgb" / f"{row['sample_id']}.png").relative_to(output_dir)).replace("\\", "/")
        result["depth_path_pattern"] = f"depth/{row['sample_id']}_depth_####.exr"
        result["mask_path_pattern"] = f"mask/{row['sample_id']}_mask_####.png"
        result["rendered_with_physical_event"] = bool(fire_object)
        rendered.append(result)
    write_jsonl(output_dir / "blender_manifest.jsonl", rendered)
    (output_dir / "renderer_info.json").write_text(
        json.dumps(
            {
                "renderer": "Blender Eevee",
                "source_dataset": str(dataset),
                "split": args.split,
                "records": len(rendered),
                "note": "Metric labels and camera matrices are copied from the source manifest; RGB/depth/mask are rendered outputs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rendered": len(rendered)}, indent=2))


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    render_rows(parse_args())
