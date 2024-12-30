# Register the custom environment
from gymnasium.envs.registration import register

CUSTOM_ENV_NAMES = [
    ('UnlockPickup-RewardShaping-v0', 'custom_envs.unlockpickup:RewardShapingUP'),   
    ('UnlockPickup-RewardShaping-Restrictive-v0', 'custom_envs.unlockpickup:RewardShapingUPExtraRewardOnce'),   
    ('UnlockPickup-RewardShaping-Simple-v0', 'custom_envs.unlockpickup:RewardShapingUPSimple'),   
    ('UnlockPickup-RewardShaping-Simple-v1', 'custom_envs.unlockpickup:RewardShapingUPSimpleV1'),   
    ('BlockedUnlockPickup-RewardShaping-v0', 'custom_envs.blockedunlockpickup:RewardShapingBUP'),
]

for env_id, entry_point in CUSTOM_ENV_NAMES:
    register(
        id=env_id,
        entry_point=entry_point
    )

    print(f"Successfully registered - {env_id}")