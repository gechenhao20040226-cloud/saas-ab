import numpy as np
import pandas as pd
from scipy import stats


METRIC_NAME_MAP = {
    "login": "登录率",
    "view_feature": "新功能曝光率",
    "click_feature": "新功能点击率",
    "use_feature": "功能使用率",
    "complete_task": "任务完成率",
    "is_paid": "付费转化率",
    "total_amount": "ARPU / 用户平均收入",
    "paid_amount": "付费用户平均付费金额"
}


def proportion_z_test(user_level, metric_col, alpha=0.05):
    """
    对比例类指标做 z-test。

    适用于：
    - 点击率
    - 使用率
    - 付费转化率
    - 任务完成率

    原理：
    比较 treatment 组和 control 组的比例是否存在显著差异。
    """

    control_data = user_level[user_level["group"] == "control"][metric_col].dropna()
    treatment_data = user_level[user_level["group"] == "treatment"][metric_col].dropna()

    n_control = len(control_data)
    n_treatment = len(treatment_data)

    if n_control == 0 or n_treatment == 0:
        return None

    x_control = control_data.sum()
    x_treatment = treatment_data.sum()

    control_rate = x_control / n_control
    treatment_rate = x_treatment / n_treatment

    diff = treatment_rate - control_rate

    # pooled proportion，用于 z-test
    pooled_rate = (x_control + x_treatment) / (n_control + n_treatment)

    pooled_se = np.sqrt(
        pooled_rate * (1 - pooled_rate) * (1 / n_control + 1 / n_treatment)
    )

    if pooled_se == 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        z_stat = diff / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 非 pooled 标准误，用于置信区间
    ci_se = np.sqrt(
        control_rate * (1 - control_rate) / n_control
        + treatment_rate * (1 - treatment_rate) / n_treatment
    )

    z_critical = stats.norm.ppf(1 - alpha / 2)

    ci_lower = diff - z_critical * ci_se
    ci_upper = diff + z_critical * ci_se

    if control_rate == 0:
        uplift = np.nan
    else:
        uplift = diff / control_rate

    is_significant = False if pd.isna(p_value) else p_value < alpha

    result = {
        "metric": metric_col,
        "metric_cn": METRIC_NAME_MAP.get(metric_col, metric_col),
        "test_type": "Proportion Z-Test",
        "control_value": control_rate,
        "treatment_value": treatment_rate,
        "difference": diff,
        "uplift": uplift,
        "statistic": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "is_significant": is_significant,
        "conclusion": "显著" if is_significant else "不显著"
    }

    return result


def mean_t_test(user_level, metric_col, alpha=0.05):
    """
    对均值类指标做 Welch's t-test。

    适用于：
    - ARPU
    - 用户平均收入
    - 付费金额

    原理：
    比较 treatment 组和 control 组的均值是否存在显著差异。
    """

    control_data = user_level[user_level["group"] == "control"][metric_col].dropna()
    treatment_data = user_level[user_level["group"] == "treatment"][metric_col].dropna()

    n_control = len(control_data)
    n_treatment = len(treatment_data)

    if n_control < 2 or n_treatment < 2:
        return None

    control_mean = control_data.mean()
    treatment_mean = treatment_data.mean()

    diff = treatment_mean - control_mean

    # Welch's t-test，不要求两组方差相等
    t_stat, p_value = stats.ttest_ind(
        treatment_data,
        control_data,
        equal_var=False,
        nan_policy="omit"
    )

    control_var = control_data.var(ddof=1)
    treatment_var = treatment_data.var(ddof=1)

    se = np.sqrt(treatment_var / n_treatment + control_var / n_control)

    # Welch-Satterthwaite 自由度
    numerator = (treatment_var / n_treatment + control_var / n_control) ** 2
    denominator = (
        (treatment_var / n_treatment) ** 2 / (n_treatment - 1)
        + (control_var / n_control) ** 2 / (n_control - 1)
    )

    if denominator == 0 or se == 0:
        df = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        df = numerator / denominator
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = diff - t_critical * se
        ci_upper = diff + t_critical * se

    if control_mean == 0:
        uplift = np.nan
    else:
        uplift = diff / control_mean

    is_significant = False if pd.isna(p_value) else p_value < alpha

    result = {
        "metric": metric_col,
        "metric_cn": METRIC_NAME_MAP.get(metric_col, metric_col),
        "test_type": "Welch T-Test",
        "control_value": control_mean,
        "treatment_value": treatment_mean,
        "difference": diff,
        "uplift": uplift,
        "statistic": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "is_significant": is_significant,
        "conclusion": "显著" if is_significant else "不显著"
    }

    return result


def run_ab_tests(user_level, alpha=0.05):
    """
    一次性运行所有 A/B Test。

    比例类指标：
    - login
    - view_feature
    - click_feature
    - use_feature
    - complete_task
    - is_paid

    均值类指标：
    - total_amount
    """

    test_results = []

    # 比例类指标
    proportion_metrics = [
        "login",
        "view_feature",
        "click_feature",
        "use_feature",
        "complete_task",
        "is_paid"
    ]

    for metric in proportion_metrics:
        if metric in user_level.columns:
            result = proportion_z_test(
                user_level=user_level,
                metric_col=metric,
                alpha=alpha
            )

            if result is not None:
                test_results.append(result)

    # 均值类指标：ARPU
    mean_metrics = [
        "total_amount"
    ]

    for metric in mean_metrics:
        if metric in user_level.columns:
            result = mean_t_test(
                user_level=user_level,
                metric_col=metric,
                alpha=alpha
            )

            if result is not None:
                test_results.append(result)

    results_df = pd.DataFrame(test_results)

    return results_df


def format_test_results_for_display(results_df):
    """
    把统计检验结果转成适合 Streamlit 页面展示的格式。
    注意：这个函数只用于展示，不用于计算。
    """

    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()

    display_df = results_df.copy()

    display_df["control_value"] = display_df["control_value"].apply(lambda x: f"{x:.4f}")
    display_df["treatment_value"] = display_df["treatment_value"].apply(lambda x: f"{x:.4f}")
    display_df["difference"] = display_df["difference"].apply(lambda x: f"{x:.4f}")
    display_df["uplift"] = display_df["uplift"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2%}"
    )
    display_df["p_value"] = display_df["p_value"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
    )
    display_df["ci_lower"] = display_df["ci_lower"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
    )
    display_df["ci_upper"] = display_df["ci_upper"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
    )

    display_df = display_df.rename(columns={
        "metric_cn": "指标",
        "test_type": "检验方法",
        "control_value": "control 数值",
        "treatment_value": "treatment 数值",
        "difference": "差异",
        "uplift": "提升幅度",
        "p_value": "p-value",
        "ci_lower": "95% CI 下限",
        "ci_upper": "95% CI 上限",
        "conclusion": "结论"
    })

    display_df = display_df[
        [
            "指标",
            "检验方法",
            "control 数值",
            "treatment 数值",
            "差异",
            "提升幅度",
            "p-value",
            "95% CI 下限",
            "95% CI 上限",
            "结论"
        ]
    ]

    return display_df


def generate_launch_recommendation(results_df):
    """
    根据 A/B Test 结果生成上线建议。
    这个版本先用简单规则，后面可以继续优化。
    """

    if results_df is None or len(results_df) == 0:
        return "当前数据不足，无法生成上线建议。"

    # 找出关键指标
    paid_result = results_df[results_df["metric"] == "is_paid"]
    use_result = results_df[results_df["metric"] == "use_feature"]
    arpu_result = results_df[results_df["metric"] == "total_amount"]

    paid_good = False
    use_good = False
    arpu_bad = False

    if len(paid_result) > 0:
        row = paid_result.iloc[0]
        paid_good = (
            row["is_significant"] is True
            and row["difference"] > 0
        )

    if len(use_result) > 0:
        row = use_result.iloc[0]
        use_good = (
            row["is_significant"] is True
            and row["difference"] > 0
        )

    if len(arpu_result) > 0:
        row = arpu_result.iloc[0]
        arpu_bad = (
            row["is_significant"] is True
            and row["difference"] < 0
        )

    if paid_good and not arpu_bad:
        return (
            "实验组的付费转化率显著提升，且 ARPU 没有显著下降。"
            "从当前实验结果看，新功能具备上线价值，建议先小范围灰度上线。"
        )

    if use_good and not arpu_bad:
        return (
            "实验组的功能使用率显著提升，说明新功能对用户行为有正向影响。"
            "但付费指标还需要继续观察，建议延长实验周期或扩大样本量后再决定是否全面上线。"
        )

    if arpu_bad:
        return (
            "实验组的收入类指标出现显著下降，说明新功能可能带来负面商业影响。"
            "当前不建议全面上线，建议先复盘功能设计和用户路径。"
        )

    return (
        "当前实验组相比对照组没有表现出稳定且显著的核心指标提升。"
        "暂不建议全面上线，可以继续观察数据或重新优化新功能方案。"
    )


if __name__ == "__main__":
    from data_generator import generate_all_data
    from data_cleaning import clean_all_data
    from analysis import run_metric_analysis

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

    results_df = run_ab_tests(user_level, alpha=0.05)

    print("A/B Test 检验结果：")
    print(results_df)

    print("\n上线建议：")
    print(generate_launch_recommendation(results_df))