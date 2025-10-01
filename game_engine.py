from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROGRESS_DIR = Path('progress')
PROGRESS_DIR.mkdir(exist_ok=True)


@dataclass
class Profile:
    name: str
    xp: int = 0
    level: int = 1
    badges: List[str] = None
    history: List[Dict] = None

    def __post_init__(self):
        if self.badges is None:
            self.badges = []
        if self.history is None:
            self.history = []


def _xp_for_next_level(level: int) -> int:
    # Simple curve: 100 xp per level
    return 100 * level


def load_profile(name: str) -> Profile:
    p = PROGRESS_DIR / f"{name.lower().strip().replace(' ','_')}.json"
    if p.exists():
        data = json.loads(p.read_text())
        return Profile(**data)
    return Profile(name=name)


def save_profile(profile: Profile) -> None:
    p = PROGRESS_DIR / f"{profile.name.lower().strip().replace(' ','_')}.json"
    p.write_text(json.dumps(asdict(profile), indent=2))


def add_xp(profile: Profile, gained: int) -> Tuple[Profile, bool]:
    profile.xp += gained
    leveled = False
    while profile.xp >= _xp_for_next_level(profile.level):
        profile.xp -= _xp_for_next_level(profile.level)
        profile.level += 1
        leveled = True
    return profile, leveled


def award_badge(profile: Profile, badge: str) -> None:
    if badge not in profile.badges:
        profile.badges.append(badge)


def current_tasks(level: int) -> List[Dict]:
    # Curriculum: early focus on lines, then digits, then alphabets
    tasks: List[Dict] = []
    if level <= 2:
        tasks += [
            {"type": "line", "line_type": "sleeping", "threshold": 55, "xp": 15, "title": "Draw a straight sleeping line"},
            {"type": "line", "line_type": "slanting", "threshold": 55, "xp": 15, "title": "Draw a neat slanting line"},
        ]
    if 2 <= level <= 5:
        tasks += [
            {"type": "digit", "digit": d, "threshold": 60, "xp": 20, "title": f"Write digit {d}"} for d in [0,1,2,3,4]
        ]
    if level >= 5:
        tasks += [
            {"type": "digit", "digit": d, "threshold": 70, "xp": 20, "title": f"Write digit {d}"} for d in [5,6,7,8,9]
        ]
    if level >= 6:
        tasks += [
            {"type": "alphabet", "letter": ch, "threshold": 0.60, "xp": 25, "title": f"Write letter {ch}"} for ch in ["A","B","C"]
        ]
    return tasks


def evaluate_task(task: Dict, result: Dict) -> Tuple[bool, int, Dict]:
    # result is output from score_line / score_digit / BN infer
    passed = False
    gained_xp = 0
    details: Dict = {}
    if task["type"] == "line":
        score = result.get("quality_score", 0)
        passed = score >= task.get("threshold", 60)
        gained_xp = task.get("xp", 10) if passed else 0
        details = {"score": score}
    elif task["type"] == "digit":
        score = result.get("quality_score", 0)
        pred = result.get("predicted_digit")
        conf = result.get("prediction_confidence", 0.0)
        # require both quality threshold and correct prediction with conf >= 0.5
        passed = (score >= task.get("threshold", 60)) and (pred == task["digit"]) and (conf >= 0.5)
        gained_xp = task.get("xp", 15) if passed else 0
        details = {"score": score, "pred": pred, "conf": conf}
    else:  # alphabet
        top = result.get("top", {})
        best_letter = None
        best_conf = 0.0
        if isinstance(top, dict) and top:
            best_letter, best_conf = next(iter(sorted(top.items(), key=lambda kv: kv[1], reverse=True)))
        passed = (best_letter == task.get("letter")) and (best_conf >= task.get("threshold", 0.6))
        gained_xp = task.get("xp", 20) if passed else 0
        details = {"best": best_letter, "conf": best_conf}
    return passed, gained_xp, details


def on_task_completed(profile: Profile, task: Dict, passed: bool, gained_xp: int, details: Dict) -> Tuple[Profile, str]:
    msg = ""
    entry = {"task": task, "passed": passed, "xp": gained_xp, "details": details}
    profile.history.append(entry)
    if passed:
        profile, leveled = add_xp(profile, gained_xp)
        msg = f"Great job! +{gained_xp} XP."
        if leveled:
            msg += f" You reached Level {profile.level}!"
        # Simple badge rules
        if task["type"] == "line" and task.get("line_type") == "sleeping" and details.get("score",0) >= 80:
            award_badge(profile, "Straight-Line Star")
        if task["type"] == "digit" and details.get("score",0) >= 85:
            award_badge(profile, "Digit Dynamo")
        if task["type"] == "alphabet" and details.get("conf",0) >= 0.85:
            award_badge(profile, "Alphabet Ace")
    else:
        msg = "Nice try! Check tips and try again."
    save_profile(profile)
    return profile, msg
