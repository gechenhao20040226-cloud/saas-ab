import pandas as pd
import numpy as np


EVENT_TYPES = [
    "login",
    "view_feature",
    "click_feature",
    "use_feature",
    "complete_task"
]


METRIC_NAME_MAP = {
    "user_count": "用户数",
    "login_rate": "登录率",
    "view_feature_rate": "新功能曝光率",
    "click_feature_rate": "新功能点击率",
    "use_feature_rate": "功能使用率",
    "complete_task_rate": "任务完成率",
    "payment_conversion_rate": "付费转化率",
    "arpu": "ARPU",
    "avg_paid_amount": "付费用户平均付费金额",
    "total_revenue": "总收入"
}


def build_user_level_data(users, assignment, events, payments):
    """
    构造用户粒度分析表。

    每一行代表一个用户，包含：
    - 用户基础信息
    - 实验分组
    - 是否发生过各类行为事件
    - 是否付费
    - 付费金额
    """

    # 用户基础信息 + 实验分组
    user_level = assignment.merge(
        users,
        on="user_id",
        how="left"
    )

    # =========================
    # 1. 处理用户行为事件
    # =========================
    if events is not None and len(events) > 0:
        event_flags = (
            events
            .drop_duplicates(subset=["user_id", "event_type"])
            .assign(flag=1)
            .pivot_table(
                index="user_id",
                columns="event_type",
                values="flag",
                aggfunc="max",
                fill_value=0
            )
            .reset_index()
        )

        # 确保所有事件列都存在
        for event in EVENT_TYPES:
            if event not in event_flags.columns:
                event_flags[event] = 0

        event_flags = event_flags[["user_id"] + EVENT_TYPES]

    else:
        event_flags = pd.DataFrame({
            "user_id": user_level["user_id"]
        })

        for event in EVENT_TYPES:
            event_flags[event] = 0

    user_level = user_level.merge(
        event_flags,
        on="user_id",
        how="left"
    )

    # 行为缺失填 0
    for event in EVENT_TYPES:
        user_level[event] = user_level[event].fillna(0).astype(int)

    # =========================
    # 2. 处理付费数据
    # =========================
    if payments is not None and len(payments) > 0:
        payment_agg = (
            payments
            .groupby("user_id", as_index=False)
            .agg(
                total_amount=("amount", "sum"),
                payment_count=("amount", "count")
            )
        )

        payment_agg["is_paid"] = 1

    else:
        payment_agg = pd.DataFrame({
            "user_id": user_level["user_id"],
            "total_amount": 0,
            "payment_count": 0,
            "is_paid": 0
        })

    user_level = user_level.merge(
        payment_agg,
        on="user_id",
        how="left"
    )

    user_level["total_amount"] = user_level["total_amount"].fillna(0)
    user_level["payment_count"] = user_level["payment_count"].fillna(0)
    user_level["is_paid"] = user_level["is_paid"].fillna(0).astype(int)

    return user_level


def calculate_group_metrics(user_level):
    """
    按 control / treatment 计算核心指标。
    """

    result = []

    for group_name, df in user_level.groupby("group"):
        user_count = len(df)

        if user_count == 0:
            continue

        paid_users = df[df["is_paid"] == 1]

        metrics = {
            "group": group_name,
            "user_count": user_count,

            # 行为类指标
            "login_rate": df["login"].mean(),
            "view_feature_rate": df["view_feature"].mean(),
            "click_feature_rate": df["click_feature"].mean(),
            "use_feature_rate": df["use_feature"].mean(),
            "complete_task_rate": df["complete_task"].mean(),

            # 付费类指标
            "payment_conversion_rate": df["is_paid"].mean(),
            "arpu": df["total_amount"].sum() / user_count,
            "avg_paid_amount": paid_users["total_amount"].mean() if len(paid_users) > 0 else 0,
            "total_revenue": df["total_amount"].sum()
        }

        result.append(metrics)

    metrics_df = pd.DataFrame(result)

    # 固定排序：control 在前，treatment 在后
    group_order = {"control": 0, "treatment": 1}
    metrics_df["group_order"] = metrics_df["group"].map(group_order)
    metrics_df = metrics_df.sort_values("group_order").drop(columns=["group_order"])

    return metrics_df


def calculate_uplift(metrics_df):
    """
    计算 treatment 相比 control 的提升幅度。

    uplift = (treatment - control) / control
    """

    if "control" not in metrics_df["group"].values or "treatment" not in metrics_df["group"].values:
        return pd.DataFrame()

    control = metrics_df[metrics_df["group"] == "control"].iloc[0]
    treatment = metrics_df[metrics_df["group"] == "treatment"].iloc[0]

    metrics_to_compare = [
        "login_rate",
        "view_feature_rate",
        "click_feature_rate",
        "use_feature_rate",
        "complete_task_rate",
        "payment_conversion_rate",
        "arpu",
        "avg_paid_amount"
    ]

    uplift_rows = []

    for metric in metrics_to_compare:
        control_value = control[metric]
        treatment_value = treatment[metric]

        absolute_diff = treatment_value - control_value

        if control_value == 0:
            uplift_percent = np.nan
        else:
            uplift_percent = absolute_diff / control_value

        uplift_rows.append({
            "metric": metric,
            "metric_cn": METRIC_NAME_MAP.get(metric, metric),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_diff": absolute_diff,
            "uplift_percent": uplift_percent
        })

    uplift_df = pd.DataFrame(uplift_rows)

    return uplift_df


def format_metrics_for_display(metrics_df):
    """
    把核心指标表转成适合页面展示的格式。
    注意：这个函数只用于展示，不用于后续计算。
    """

    display_df = metrics_df.copy()

    percent_cols = [
        "login_rate",
        "view_feature_rate",
        "click_feature_rate",
        "use_feature_rate",
        "complete_task_rate",
        "payment_conversion_rate"
    ]

    money_cols = [
        "arpu",
        "avg_paid_amount",
        "total_revenue"
    ]

    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

    for col in money_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    display_df = display_df.rename(columns=METRIC_NAME_MAP)
    display_df = display_df.rename(columns={"group": "实验组"})

    return display_df


def format_uplift_for_display(uplift_df):
    """
    把 uplift 表转成适合页面展示的格式。
    """

    display_df = uplift_df.copy()

    if len(display_df) == 0:
        return display_df

    display_df["control_value"] = display_df["control_value"].apply(lambda x: f"{x:.4f}")
    display_df["treatment_value"] = display_df["treatment_value"].apply(lambda x: f"{x:.4f}")
    display_df["absolute_diff"] = display_df["absolute_diff"].apply(lambda x: f"{x:.4f}")
    display_df["uplift_percent"] = display_df["uplift_percent"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2%}"
    )

    display_df = display_df.rename(columns={
        "metric_cn": "指标",
        "control_value": "control 数值",
        "treatment_value": "treatment 数值",
        "absolute_diff": "绝对差异",
        "uplift_percent": "提升幅度"
    })

    display_df = display_df[
        ["指标", "control 数值", "treatment 数值", "绝对差异", "提升幅度"]
    ]

    return display_df


def run_metric_analysis(users, assignment, events, payments):
    """
    一键完成指标分析。

    返回：
    - user_level：用户粒度明细表
    - metrics_df：分组核心指标表
    - uplift_df：提升幅度表
    """

    user_level = build_user_level_data(
        users,
        assignment,
        events,
        payments
    )

    metrics_df = calculate_group_metrics(user_level)

    uplift_df = calculate_uplift(metrics_df)

    return user_level, metrics_df, uplift_df


if __name__ == "__main__":
    from data_generator import generate_all_data
    from data_cleaning import clean_all_data

    users, assignment, events, payments = generate_all_data(n_users=5000)

    (
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean,
        cleaning_report
    ) = clean_all_data(users, assignment, events, payments)

    user_level, metrics_df, uplift_df = run_metric_analysis(
        users_clean,
        assignment_clean,
        events_clean,
        payments_clean
    )

    print("用户粒度数据：")
    print(user_level.head())

    print("\n核心指标：")
    print(metrics_df)

    print("\nUplift 分析：")
    print(uplift_df)