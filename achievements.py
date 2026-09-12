from dataclasses import dataclass


@dataclass(frozen=True)
class Achievement:
    id: str
    name: str
    description: str


ACHIEVEMENTS = [
    Achievement("first_blink", "First Blink", "Recorded your first blink."),
    Achievement("blink_streak", "Blink Streak", "Recorded ten blinks in one session."),
]


def unlocked_achievements(stats, performance):
    unlocked = []
    if stats.total_blinks >= 1:
        unlocked.append(ACHIEVEMENTS[0])
    if stats.total_blinks >= 10:
        unlocked.append(ACHIEVEMENTS[1])
    return unlocked