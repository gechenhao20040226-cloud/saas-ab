import pandas as pd
import numpy as np


def clean_users(users):
    """
    清洗用户表
    """
    users = users.copy()

    before_rows = len(users)

    # 去除完全重复行
    users = users.drop_duplicates()

    # user_id 转成数字
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")

    # 删除 user_id 缺失的行
    users = users.dropna(subset=["user_id"])

    # user_id 转成整数
    users["user_id"] = users["user_id"].astype(int)

    # 同一个 user_id 只保留第一条
    users = users.drop_duplicates(subset=["user_id"], keep="first")

    # 日期格式统一
    users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce")

    # 缺失值处理
    users["region"] = users["region"].fillna("未知地区")
    users["device"] = users["device"].fillna("未知设备")
    users["user_type"] = users["user_type"].fillna("unknown_user")

    after_rows = len(users)

    report = {
        "table": "users",
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": before_rows - after_rows
    }

    return users, report


def clean_assignment(assignment, valid_user_ids):
    """
    清洗实验分组表
    """
    assignment = assignment.copy()

    before_rows = len(assignment)

    assignment = assignment.drop_duplicates()

    assignment["user_id"] = pd.to_numeric(assignment["user_id"], errors="coerce")
    assignment = assignment.dropna(subset=["user_id"])
    assignment["user_id"] = assignment["user_id"].astype(int)

    # 只保留 users 表中存在的 user_id
    assignment = assignment[assignment["user_id"].isin(valid_user_ids)]

    # 只保留合法实验组
    assignment = assignment[assignment["group"].isin(["control", "treatment"])]

    # 一个用户只能属于一个实验组
    assignment = assignment.drop_duplicates(subset=["user_id"], keep="first")

    # 日期格式统一
    assignment["assigned_at"] = pd.to_datetime(
        assignment["assigned_at"],
        errors="coerce"
    )

    after_rows = len(assignment)

    report = {
        "table": "experiment_assignment",
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": before_rows - after_rows
    }

    return assignment, report


def clean_events(events, valid_user_ids):
    """
    清洗用户行为事件表
    """
    events = events.copy()

    before_rows = len(events)

    events = events.drop_duplicates()

    events["user_id"] = pd.to_numeric(events["user_id"], errors="coerce")
    events = events.dropna(subset=["user_id"])
    events["user_id"] = events["user_id"].astype(int)

    # 只保留合法用户
    events = events[events["user_id"].isin(valid_user_ids)]

    # 统一时间格式
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")

    # 删除事件时间为空的行
    events = events.dropna(subset=["event_time"])

    # 只保留合法事件类型
    valid_event_types = [
        "login",
        "view_feature",
        "click_feature",
        "use_feature",
        "complete_task"
    ]

    events = events[events["event_type"].isin(valid_event_types)]

    after_rows = len(events)

    report = {
        "table": "events",
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": before_rows - after_rows
    }

    return events, report


def clean_payments(payments, valid_user_ids):
    """
    清洗付费表
    """
    payments = payments.copy()

    before_rows = len(payments)

    payments = payments.drop_duplicates()

    payments["user_id"] = pd.to_numeric(payments["user_id"], errors="coerce")
    payments = payments.dropna(subset=["user_id"])
    payments["user_id"] = payments["user_id"].astype(int)

    # 只保留合法用户
    payments = payments[payments["user_id"].isin(valid_user_ids)]

    # 统一时间格式
    payments["payment_time"] = pd.to_datetime(
        payments["payment_time"],
        errors="coerce"
    )

    # amount 转成数字
    payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce")

    # 删除金额缺失、金额小于等于 0、金额异常过大的数据
    payments = payments.dropna(subset=["amount"])
    payments = payments[(payments["amount"] > 0) & (payments["amount"] <= 1000)]

    # 统一订阅状态
    payments["subscription_status"] = payments["subscription_status"].fillna("paid")

    after_rows = len(payments)

    report = {
        "table": "payments",
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": before_rows - after_rows
    }

    return payments, report


def clean_all_data(users, assignment, events, payments):
    """
    一次性清洗四张表
    返回：
    - clean_users_df
    - clean_assignment_df
    - clean_events_df
    - clean_payments_df
    - cleaning_report
    """

    users_clean, users_report = clean_users(users)

    valid_user_ids = set(users_clean["user_id"])

    assignment_clean, assignment_report = clean_assignment(
        assignment,
        valid_user_ids
    )

    # 只保留同时存在于 users 和 assignment 的用户
    valid_experiment_user_ids = set(assignment_clean["user_id"])

    events_clean, events_report = clean_events(
        events,
        valid_experiment_user_ids
    )

    payments_clean, payments_report = clean_payments(
        payments,
        valid_experiment_user_ids
    )

    cleaning_report = pd.DataFrame([
        users_report,
        assignment_report,
        events_report,
        payments_report
    ])

    return (
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean,
        cleaning_report
    )


def load_sample_data(data_path="sample_data"):
    """
    从 sample_data 文件夹读取四张表
    """
    users = pd.read_csv(f"{data_path}/users.csv")
    assignment = pd.read_csv(f"{data_path}/experiment_assignment.csv")
    events = pd.read_csv(f"{data_path}/events.csv")
    payments = pd.read_csv(f"{data_path}/payments.csv")

    return users, assignment, events, payments


if __name__ == "__main__":
    from data_generator import generate_all_data

    users, assignment, events, payments = generate_all_data(n_users=5000)

    (
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean,
        cleaning_report
    ) = clean_all_data(users, assignment, events, payments)

    print("数据清洗完成！")
    print(cleaning_report)

    print("users_clean:", users_clean.shape)
    print("assignment_clean:", assignment_clean.shape)
    print("events_clean:", events_clean.shape)
    print("payments_clean:", payments_clean.shape)