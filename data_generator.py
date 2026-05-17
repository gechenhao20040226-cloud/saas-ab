import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def clip_prob(x):
    """把概率限制在 0 到 0.95 之间，避免出现超过 100% 的概率。"""
    return min(max(x, 0), 0.95)


def get_baseline_config(baseline_level):
    """
    行为漏斗基线：
    控制旧版本产品原本的转化水平。
    """

    configs = {
        "low": {
            "login": 0.60,
            "view_given_login": 0.45,
            "click_given_view": 0.35,
            "use_given_click": 0.45,
            "complete_given_use": 0.55,
            "pay": 0.03
        },
        "normal": {
            "login": 0.75,
            "view_given_login": 0.70,
            "click_given_view": 0.50,
            "use_given_click": 0.65,
            "complete_given_use": 0.70,
            "pay": 0.08
        },
        "high": {
            "login": 0.85,
            "view_given_login": 0.80,
            "click_given_view": 0.65,
            "use_given_click": 0.75,
            "complete_given_use": 0.80,
            "pay": 0.12
        }
    }

    return configs.get(baseline_level, configs["normal"])


def get_effect_config(effect_level):
    """
    效应量：
    控制 treatment 新功能组相比 control 旧版本组提升多少。
    """

    configs = {
        "no_effect": {
            "view": 1.00,
            "click": 1.00,
            "use": 1.00,
            "complete": 1.00,
            "pay": 1.00,
            "amount": 1.00
        },
        "slight": {
            "view": 1.05,
            "click": 1.10,
            "use": 1.12,
            "complete": 1.10,
            "pay": 1.08,
            "amount": 1.00
        },
        "medium": {
            "view": 1.12,
            "click": 1.25,
            "use": 1.35,
            "complete": 1.30,
            "pay": 1.30,
            "amount": 1.05
        },
        "strong": {
            "view": 1.25,
            "click": 1.50,
            "use": 1.65,
            "complete": 1.60,
            "pay": 1.55,
            "amount": 1.10
        },
        "negative": {
            "view": 1.05,
            "click": 1.00,
            "use": 0.90,
            "complete": 0.85,
            "pay": 0.80,
            "amount": 0.90
        }
    }

    return configs.get(effect_level, configs["medium"])


def get_noise_config(noise_level):
    """
    噪声水平：
    控制数据脏乱程度，包括缺失、异常、重复、实验污染。
    """

    configs = {
        "clean": {
            "duplicate_user_rate": 0.000,
            "missing_user_rate": 0.000,
            "duplicate_event_rate": 0.000,
            "abnormal_payment_rate": 0.000,
            "missing_payment_rate": 0.000,
            "pollution_rate": 0.000
        },
        "normal": {
            "duplicate_user_rate": 0.002,
            "missing_user_rate": 0.001,
            "duplicate_event_rate": 0.003,
            "abnormal_payment_rate": 0.010,
            "missing_payment_rate": 0.006,
            "pollution_rate": 0.030
        },
        "dirty": {
            "duplicate_user_rate": 0.010,
            "missing_user_rate": 0.008,
            "duplicate_event_rate": 0.015,
            "abnormal_payment_rate": 0.040,
            "missing_payment_rate": 0.020,
            "pollution_rate": 0.100
        }
    }

    return configs.get(noise_level, configs["normal"])


def generate_users(n_users=5000, seed=42):
    np.random.seed(seed)

    user_ids = np.arange(1, n_users + 1)

    regions = np.random.choice(
        ["华东", "华北", "华南", "西南", "海外"],
        size=n_users,
        p=[0.35, 0.25, 0.20, 0.15, 0.05]
    )

    devices = np.random.choice(
        ["iOS", "Android", "Web"],
        size=n_users,
        p=[0.35, 0.40, 0.25]
    )

    user_types = np.random.choice(
        ["new_user", "old_user"],
        size=n_users,
        p=[0.65, 0.35]
    )

    start_date = datetime(2026, 1, 1)
    signup_dates = [
        start_date + timedelta(days=int(x))
        for x in np.random.randint(0, 90, size=n_users)
    ]

    users = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup_dates,
        "region": regions,
        "device": devices,
        "user_type": user_types
    })

    return users


def generate_experiment_assignment(users, seed=42):
    np.random.seed(seed)

    assignment = users[["user_id"]].copy()

    assignment["group"] = np.random.choice(
        ["control", "treatment"],
        size=len(users),
        p=[0.5, 0.5]
    )

    base_date = datetime(2026, 4, 1)
    assignment["assigned_at"] = [
        base_date + timedelta(days=int(x))
        for x in np.random.randint(0, 7, size=len(users))
    ]

    return assignment


def build_exposure_map(assignment, noise_level="normal", seed=42):
    """
    实验污染：
    有一小部分用户实际看到的版本和分组不一致。
    比如 control 用户误看到新功能，或者 treatment 用户没有真正看到新功能。
    """

    np.random.seed(seed)

    noise = get_noise_config(noise_level)
    pollution_rate = noise["pollution_rate"]

    exposure_map = {}

    for _, row in assignment.iterrows():
        user_id = row["user_id"]
        assigned_group = row["group"]

        if np.random.rand() < pollution_rate:
            actual_group = "treatment" if assigned_group == "control" else "control"
        else:
            actual_group = assigned_group

        exposure_map[user_id] = actual_group

    return exposure_map


def generate_events(
    users,
    assignment,
    baseline_level="normal",
    effect_level="medium",
    noise_level="normal",
    seed=42
):
    """
    生成用户行为事件表。

    采用漏斗逻辑：
    login -> view_feature -> click_feature -> use_feature -> complete_task
    """

    np.random.seed(seed)

    baseline = get_baseline_config(baseline_level)
    effect = get_effect_config(effect_level)
    exposure_map = build_exposure_map(assignment, noise_level=noise_level, seed=seed + 100)

    events = []

    merged = users.merge(assignment, on="user_id", how="left")

    for _, row in merged.iterrows():
        user_id = row["user_id"]
        assigned_group = row["group"]
        actual_group = exposure_map.get(user_id, assigned_group)

        login_prob = baseline["login"]
        view_prob = baseline["view_given_login"]
        click_prob = baseline["click_given_view"]
        use_prob = baseline["use_given_click"]
        complete_prob = baseline["complete_given_use"]

        if actual_group == "treatment":
            view_prob = clip_prob(view_prob * effect["view"])
            click_prob = clip_prob(click_prob * effect["click"])
            use_prob = clip_prob(use_prob * effect["use"])
            complete_prob = clip_prob(complete_prob * effect["complete"])

        event_base_time = datetime(2026, 4, 1)

        def random_event_time():
            return event_base_time + timedelta(
                days=int(np.random.randint(0, 30)),
                hours=int(np.random.randint(0, 24)),
                minutes=int(np.random.randint(0, 60))
            )

        if np.random.rand() < login_prob:
            events.append([user_id, random_event_time(), "login"])

            if np.random.rand() < view_prob:
                events.append([user_id, random_event_time(), "view_feature"])

                if np.random.rand() < click_prob:
                    events.append([user_id, random_event_time(), "click_feature"])

                    if np.random.rand() < use_prob:
                        events.append([user_id, random_event_time(), "use_feature"])

                        if np.random.rand() < complete_prob:
                            events.append([user_id, random_event_time(), "complete_task"])

    events_df = pd.DataFrame(
        events,
        columns=["user_id", "event_time", "event_type"]
    )

    return events_df


def generate_payments(
    users,
    assignment,
    baseline_level="normal",
    effect_level="medium",
    noise_level="normal",
    seed=42
):
    """
    生成付费数据表。
    treatment 组付费概率和付费金额由效应量控制。
    """

    np.random.seed(seed)

    baseline = get_baseline_config(baseline_level)
    effect = get_effect_config(effect_level)
    exposure_map = build_exposure_map(assignment, noise_level=noise_level, seed=seed + 100)

    payments = []

    merged = users.merge(assignment, on="user_id", how="left")

    for _, row in merged.iterrows():
        user_id = row["user_id"]
        assigned_group = row["group"]
        actual_group = exposure_map.get(user_id, assigned_group)

        pay_prob = baseline["pay"]
        amount_multiplier = 1.00

        if actual_group == "treatment":
            pay_prob = clip_prob(pay_prob * effect["pay"])
            amount_multiplier = effect["amount"]

        if np.random.rand() < pay_prob:
            payment_time = datetime(2026, 4, 1) + timedelta(
                days=int(np.random.randint(0, 30)),
                hours=int(np.random.randint(0, 24))
            )

            base_amount = np.random.choice(
                [29, 49, 99, 199],
                p=[0.45, 0.30, 0.20, 0.05]
            )

            amount = round(base_amount * amount_multiplier, 2)

            payments.append([
                user_id,
                payment_time,
                amount,
                "paid"
            ])

    payments_df = pd.DataFrame(
        payments,
        columns=["user_id", "payment_time", "amount", "subscription_status"]
    )

    return payments_df


def add_dirty_data(users, assignment, events, payments, noise_level="normal", seed=42):
    """
    加入数据噪声：
    - 重复用户
    - 缺失地区
    - 重复事件
    - 异常金额
    - 缺失金额
    """

    np.random.seed(seed)

    noise = get_noise_config(noise_level)

    # users 重复行
    duplicate_user_n = int(len(users) * noise["duplicate_user_rate"])
    if duplicate_user_n > 0:
        duplicate_users = users.sample(duplicate_user_n, random_state=seed)
        users = pd.concat([users, duplicate_users], ignore_index=True)

    # users 缺失地区
    missing_user_n = int(len(users) * noise["missing_user_rate"])
    if missing_user_n > 0:
        missing_index = users.sample(missing_user_n, random_state=seed + 1).index
        users.loc[missing_index, "region"] = np.nan

    # events 重复行
    duplicate_event_n = int(len(events) * noise["duplicate_event_rate"])
    if duplicate_event_n > 0 and len(events) > duplicate_event_n:
        duplicate_events = events.sample(duplicate_event_n, random_state=seed)
        events = pd.concat([events, duplicate_events], ignore_index=True)

    # payments 异常金额
    abnormal_payment_n = int(len(payments) * noise["abnormal_payment_rate"])
    if abnormal_payment_n > 0 and len(payments) > abnormal_payment_n:
        abnormal_index = payments.sample(abnormal_payment_n, random_state=seed).index

        abnormal_values = np.random.choice(
            [-99, 0, 99999],
            size=abnormal_payment_n
        )

        payments.loc[abnormal_index, "amount"] = abnormal_values

    # payments 缺失金额
    missing_payment_n = int(len(payments) * noise["missing_payment_rate"])
    if missing_payment_n > 0 and len(payments) > missing_payment_n:
        missing_pay_index = payments.sample(missing_payment_n, random_state=seed + 2).index
        payments.loc[missing_pay_index, "amount"] = np.nan

    return users, assignment, events, payments


def generate_all_data(
    n_users=5000,
    save_path="sample_data",
    seed=42,
    baseline_level="normal",
    effect_level="medium",
    noise_level="normal"
):
    """
    一次性生成四张表，并保存到 sample_data 文件夹。
    """

    users = generate_users(n_users=n_users, seed=seed)
    assignment = generate_experiment_assignment(users, seed=seed)

    events = generate_events(
        users,
        assignment,
        baseline_level=baseline_level,
        effect_level=effect_level,
        noise_level=noise_level,
        seed=seed
    )

    payments = generate_payments(
        users,
        assignment,
        baseline_level=baseline_level,
        effect_level=effect_level,
        noise_level=noise_level,
        seed=seed
    )

    users, assignment, events, payments = add_dirty_data(
        users,
        assignment,
        events,
        payments,
        noise_level=noise_level,
        seed=seed
    )

    os.makedirs(save_path, exist_ok=True)

    users.to_csv(os.path.join(save_path, "users.csv"), index=False, encoding="utf-8-sig")
    assignment.to_csv(os.path.join(save_path, "experiment_assignment.csv"), index=False, encoding="utf-8-sig")
    events.to_csv(os.path.join(save_path, "events.csv"), index=False, encoding="utf-8-sig")
    payments.to_csv(os.path.join(save_path, "payments.csv"), index=False, encoding="utf-8-sig")

    return users, assignment, events, payments


if __name__ == "__main__":
    users, assignment, events, payments = generate_all_data(
        n_users=5000,
        baseline_level="normal",
        effect_level="medium",
        noise_level="normal"
    )

    print("模拟实验数据已生成！")
    print("users:", users.shape)
    print("assignment:", assignment.shape)
    print("events:", events.shape)
    print("payments:", payments.shape)