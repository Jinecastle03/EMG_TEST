"""Pose registry. The default 5 poses + Rest; users can add new poses (stored in config/poses.json).

`name` is the label stored in the data (English, never changes); `ko` is the Korean display name.
A model is always trained on the pose labels that exist in the selected sessions, so a newly added pose
only needs to be recorded (Collect Data) and trained (Train) - no code change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import config as C

LOW_HINT = "가볍게 — 최대 힘의 약 20–30 %"
HIGH_HINT = "강하게 — 무리하지 않는 선에서 최대 힘의 약 60–70 %"


@dataclass
class Pose:
    name: str
    instruction: str = ""
    obj: str = ""
    has_effort: bool = True
    low_hint: str = LOW_HINT
    high_hint: str = HIGH_HINT
    builtin: bool = False
    tags: list = field(default_factory=list)
    ko: str = ""                 # Korean display name
    key_point: str = ""          # what distinguishes it from similar poses


DEFAULT_POSES = [
    Pose(C.REST, "손과 팔의 힘을 완전히 빼고, 동작할 때와 같은 팔 자세(팔꿈치 약 90°, 엄지가 위)를 유지합니다.",
         "물체 없음", has_effort=False, builtin=True, ko="휴식",
         key_point="아무 힘도 주지 않기"),
    Pose("Power Grip", "물병을 손바닥 전체와 네 손가락으로 감싸 쥡니다. 엄지는 반대편에서 감쌉니다. 주먹 쥐듯이 꽉 잡는 동작입니다.",
         "물병 (지름 4–5 cm 원통)", builtin=True, tags=["grip"], ko="주먹 쥐기",
         key_point="손바닥이 물체에 닿음 · 다섯 손가락 모두 사용"),
    Pose("Precision Pinch", "엄지 끝과 검지 끝을 맞대고 그 사이에 동전을 집습니다(OK 사인 모양). 중지·약지·새끼는 펴서 힘을 뺍니다.",
         "동전 / 작은 블록", builtin=True, tags=["pinch"], ko="손끝 집기 (OK 모양)",
         key_point="엄지 '끝' ↔ 검지 '끝' · 손바닥은 물체에 안 닿음"),
    Pose("Key Pinch", "검지를 구부려 옆면을 만들고, 엄지 지문면으로 그 옆면에 카드를 눌러 집습니다. 열쇠를 꽂아 돌릴 때의 손 모양입니다.",
         "카드 / 열쇠", builtin=True, tags=["pinch"], ko="옆으로 집기 (열쇠 잡기)",
         key_point="엄지 ↔ 검지 '옆면' (손끝 집기와 다름) · 나머지 손가락은 말아 둠"),
    Pose("Index Press", "검지만 펴서 손끝으로 책상을 아래로 누릅니다(초인종 누르듯). 다른 손가락은 주먹 쥐듯 말아 두고, 팔과 손목은 움직이지 않습니다.",
         "책상 표면 / 버튼", builtin=True, tags=["press"], ko="검지로 누르기",
         key_point="검지 하나만 힘 · 물체를 집지 않음"),
    Pose("Twist CW", "왼손으로 병을 잡고, 오른손으로 뚜껑을 쥐어 위에서 볼 때 시계 방향(잠그는 방향)으로 돌립니다. "
         "Low/High는 속도가 아니라 쥐고 돌리는 힘으로 구분하고, 속도는 항상 같게(1초에 한 번) 합니다.",
         "뚜껑 달린 병", builtin=True, tags=["twist", "dynamic"], ko="뚜껑 돌리기 (시계 방향)",
         key_point="손목/전완이 회전함 · 속도는 일정하게"),
]

# Korean names for labels of the legacy session (not in the default registry)
LEGACY_KO = {"Grip": "쥐기 (기존)", "Twist CW": "시계 방향 돌리기", "Twist CCW": "반시계 방향 돌리기"}


def poses_path():
    return C.CONFIG_DIR / "poses.json"


def load_poses() -> list[Pose]:
    p = poses_path()
    if not p.exists():
        return list(DEFAULT_POSES)
    try:
        raw = C.load_json(p)
        stored = [Pose(**{k: v for k, v in d.items() if k in Pose.__dataclass_fields__}) for d in raw]
    except Exception:
        return list(DEFAULT_POSES)
    # built-in poses always use the current default text; user-added poses are kept as stored
    out = list(DEFAULT_POSES)
    names = {q.name for q in out}
    out += [q for q in stored if not q.builtin and q.name not in names]
    return out


def save_poses(poses: list[Pose]):
    C.save_json([asdict(p) for p in poses if not p.builtin], poses_path())


def add_pose(pose: Pose) -> list[Pose]:
    poses = load_poses()
    if any(p.name.lower() == pose.name.lower() for p in poses):
        raise ValueError(f"Pose '{pose.name}' already exists")
    if not pose.name.strip() or "|" in pose.name or "#" in pose.name:
        raise ValueError("Pose name must be non-empty and must not contain '|' or '#'")
    poses.append(pose)
    save_poses(poses)
    return poses


def get_pose(name: str) -> Pose | None:
    for p in load_poses():
        if p.name == name:
            return p
    return None


def ko_name(name: str) -> str:
    p = get_pose(name)
    if p is not None and p.ko:
        return p.ko
    return LEGACY_KO.get(name, "")
