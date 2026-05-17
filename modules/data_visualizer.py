"""
=============================================================
  modules/data_visualizer.py — Charts & Analytics
=============================================================
  All Matplotlib-based charts are generated here.
  Functions return Matplotlib Figure objects that Streamlit
  can display directly with st.pyplot(fig).

  Demonstrates:
    - Matplotlib (bar, pie, line, heatmap charts)
    - Pandas data manipulation for charting
    - OOP — DataVisualizer class
    - NumPy for calculations
    - Functions returning objects
=============================================================
"""

import os
import sys
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no pop-up window)
import matplotlib.pyplot      as plt
import matplotlib.ticker      as ticker
from   matplotlib.figure      import Figure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers import logger


# ─────────────────────────────────────────────
#  SHARED STYLE SETTINGS
# ─────────────────────────────────────────────
FONT_FAMILY   = "DejaVu Sans"
BG_COLOR      = "#0E1117"    # Streamlit dark background
FG_COLOR      = "#FAFAFA"    # Light text
GRID_COLOR    = "#2C2C2C"

# Apply a consistent dark style to all charts
plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : "#1A1A2E",
    "axes.edgecolor"   : GRID_COLOR,
    "axes.labelcolor"  : FG_COLOR,
    "text.color"       : FG_COLOR,
    "xtick.color"      : FG_COLOR,
    "ytick.color"      : FG_COLOR,
    "grid.color"       : GRID_COLOR,
    "legend.facecolor" : "#1A1A2E",
    "legend.edgecolor" : GRID_COLOR,
    "font.family"      : FONT_FAMILY,
})

# Project colour palette (index 0=green/present, 1=red/absent, …)
COLORS = cfg.CHART_COLORS


class DataVisualizer:
    """
    Generates data visualisation figures for the attendance dashboard.
    All public methods return a matplotlib.figure.Figure object.
    """

    # ─────────────────────────────────────────
    #  1. DAILY ATTENDANCE BAR CHART
    # ─────────────────────────────────────────

    @staticmethod
    def plot_daily_attendance(daily_df: pd.DataFrame) -> Figure:
        """
        Grouped bar chart: Present vs Absent count per day.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Must have columns: Date, Present, Absent

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        if daily_df is None or daily_df.empty:
            ax.text(0.5, 0.5, "No attendance data available.",
                    ha="center", va="center", fontsize=14)
            ax.set_axis_off()
            return fig

        dates   = daily_df["Date"].tolist()
        present = daily_df.get("Present", pd.Series([0]*len(dates))).tolist()
        absent  = daily_df.get("Absent",  pd.Series([0]*len(dates))).tolist()

        x     = np.arange(len(dates))   # NumPy array for bar positions
        width = 0.35                      # Width of each bar

        # Plot two sets of bars side by side
        bars_p = ax.bar(x - width/2, present, width,
                        label="Present", color=COLORS[0], alpha=0.85)
        bars_a = ax.bar(x + width/2, absent,  width,
                        label="Absent",  color=COLORS[1], alpha=0.85)

        # Add value labels on top of each bar
        for bar in bars_p:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        str(int(h)), ha="center", va="bottom", fontsize=9)
        for bar in bars_a:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        str(int(h)), ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Date",         fontsize=12)
        ax.set_ylabel("Student Count", fontsize=12)
        ax.set_title("Daily Attendance Overview", fontsize=15, fontweight="bold",
                     pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=9)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  2. ATTENDANCE PERCENTAGE PIE CHART
    # ─────────────────────────────────────────

    @staticmethod
    def plot_today_pie(stats: dict) -> Figure:
        """
        Pie chart showing today's present vs absent split.

        Parameters
        ----------
        stats : dict
            Keys: present, absent, total_students
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        present = stats.get("present", 0)
        absent  = stats.get("absent",  0)

        if present == 0 and absent == 0:
            ax.text(0.5, 0.5, "No data for today.",
                    ha="center", va="center", fontsize=14)
            ax.set_axis_off()
            return fig

        sizes  = [present, absent]
        labels = [f"Present\n{present}", f"Absent\n{absent}"]
        colors = [COLORS[0], COLORS[1]]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels     = labels,
            colors     = colors,
            autopct    = "%1.1f%%",    # Show percentage on each slice
            startangle = 140,          # Rotate so 'Present' starts at top
            wedgeprops = {"edgecolor": BG_COLOR, "linewidth": 2},
            textprops  = {"color": FG_COLOR, "fontsize": 11},
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")

        ax.set_title("Today's Attendance Split", fontsize=14,
                     fontweight="bold", pad=20)
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  3. STUDENT ATTENDANCE PERCENTAGE BAR
    # ─────────────────────────────────────────

    @staticmethod
    def plot_student_percentages(summary_df: pd.DataFrame) -> Figure:
        """
        Horizontal bar chart of attendance percentage per student.
        Color-coded: green ≥75%, orange 50-74%, red <50%.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Columns: Student_ID, Student_Name, Attendance_Percentage
        """
        fig, ax = plt.subplots(figsize=(10, max(4, len(summary_df) * 0.55)))

        if summary_df is None or summary_df.empty:
            ax.text(0.5, 0.5, "No data available.", ha="center", va="center",
                    fontsize=14)
            ax.set_axis_off()
            return fig

        # Sort by percentage ascending (so highest is at top after invert)
        df = summary_df.sort_values("Attendance_Percentage", ascending=True)

        names  = df["Student_ID"].tolist()
        percs  = df["Attendance_Percentage"].tolist()

        # Assign colour based on percentage threshold
        bar_colors = []
        for p in percs:
            if p >= 75:
                bar_colors.append(COLORS[0])   # Green
            elif p >= 50:
                bar_colors.append(COLORS[3])   # Orange
            else:
                bar_colors.append(COLORS[1])   # Red

        bars = ax.barh(names, percs, color=bar_colors, alpha=0.85, height=0.6)

        # Add percentage label at end of each bar
        for bar, pct in zip(bars, percs):
            ax.text(
                min(pct + 1, 101), bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%",
                va="center", fontsize=9
            )

        # Draw a dashed 75% threshold line
        ax.axvline(x=75, color=COLORS[3], linestyle="--", linewidth=1.5,
                   label="75% threshold")

        ax.set_xlim(0, 110)
        ax.set_xlabel("Attendance Percentage (%)", fontsize=12)
        ax.set_title("Student Attendance Percentage", fontsize=14,
                     fontweight="bold", pad=15)
        ax.legend(fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  4. MONTHLY TREND LINE CHART
    # ─────────────────────────────────────────

    @staticmethod
    def plot_monthly_trend(all_df: pd.DataFrame) -> Figure:
        """
        Line chart showing total 'Present' count aggregated by month.

        Parameters
        ----------
        all_df : pd.DataFrame
            Full attendance DataFrame (all dates).
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        if all_df is None or all_df.empty:
            ax.text(0.5, 0.5, "No historical data.", ha="center", va="center",
                    fontsize=14)
            ax.set_axis_off()
            return fig

        # Parse dates and extract year-month
        df = all_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)

        df["Month"] = df["Date"].dt.to_period("M")   # '2024-01', '2024-02', …

        # Count Present records per month
        monthly = (
            df[df["Status"] == "Present"]
            .groupby("Month")
            .size()
            .reset_index(name="Present_Count")
        )
        monthly["Month_str"] = monthly["Month"].astype(str)

        if monthly.empty:
            ax.text(0.5, 0.5, "No 'Present' records found.", ha="center",
                    va="center", fontsize=14)
            ax.set_axis_off()
            return fig

        ax.plot(
            monthly["Month_str"],
            monthly["Present_Count"],
            marker="o",
            linewidth=2.5,
            color=COLORS[2],
            markersize=8,
            markerfacecolor=COLORS[0],
        )

        # Fill area under the line
        ax.fill_between(
            monthly["Month_str"],
            monthly["Present_Count"],
            alpha=0.15,
            color=COLORS[2],
        )

        # Annotate each data point
        for _, row in monthly.iterrows():
            ax.text(row["Month_str"], row["Present_Count"] + 0.3,
                    str(int(row["Present_Count"])),
                    ha="center", fontsize=9)

        ax.set_xlabel("Month",           fontsize=12)
        ax.set_ylabel("Total Attendees", fontsize=12)
        ax.set_title("Monthly Attendance Trend", fontsize=14,
                     fontweight="bold", pad=15)
        ax.grid(linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  5. DEPARTMENT-WISE PIE CHART
    # ─────────────────────────────────────────

    @staticmethod
    def plot_department_distribution(students_df: pd.DataFrame) -> Figure:
        """
        Pie chart: number of students per department.

        Parameters
        ----------
        students_df : pd.DataFrame
            Columns include 'Department'.
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        if students_df is None or students_df.empty:
            ax.text(0.5, 0.5, "No student data.", ha="center", va="center",
                    fontsize=14)
            ax.set_axis_off()
            return fig

        # Count students per department using value_counts()
        dept_counts = students_df["Department"].value_counts()

        wedges, texts, autotexts = ax.pie(
            dept_counts.values,
            labels     = dept_counts.index,
            autopct    = "%1.1f%%",
            startangle = 90,
            colors     = COLORS * ((len(dept_counts) // len(COLORS)) + 1),
            wedgeprops = {"edgecolor": BG_COLOR, "linewidth": 2},
            textprops  = {"color": FG_COLOR, "fontsize": 10},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")

        ax.set_title("Students by Department", fontsize=14,
                     fontweight="bold", pad=20)
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  6. ATTENDANCE HEATMAP (student × date)
    # ─────────────────────────────────────────

    @staticmethod
    def plot_attendance_heatmap(all_df: pd.DataFrame) -> Figure:
        """
        Heatmap: rows = students, columns = dates.
        Cell is 1 (green) if Present, 0 (red) if Absent/missing.

        Parameters
        ----------
        all_df : pd.DataFrame
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        if all_df is None or all_df.empty:
            ax.text(0.5, 0.5, "No data for heatmap.", ha="center", va="center",
                    fontsize=14)
            ax.set_axis_off()
            return fig

        # Pivot table: index=Student_ID, columns=Date, values=Status
        # 1 if Present, 0 otherwise
        df = all_df.copy()
        df["is_present"] = (df["Status"] == "Present").astype(int)

        pivot = df.pivot_table(
            index   = "Student_ID",
            columns = "Date",
            values  = "is_present",
            aggfunc = "max",
            fill_value = 0,
        )

        # Limit to most recent 20 days to keep chart readable
        if pivot.shape[1] > 20:
            pivot = pivot.iloc[:, -20:]

        # imshow displays a 2D array as a grid of colours
        im = ax.imshow(pivot.values, cmap="RdYlGn",
                       aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        ax.set_title("Attendance Heatmap (Green=Present, Red=Absent)",
                     fontsize=13, fontweight="bold", pad=15)
        fig.colorbar(im, ax=ax, shrink=0.6, label="0=Absent / 1=Present")
        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────
    #  7. TOP ATTENDEES CHART
    # ─────────────────────────────────────────

    @staticmethod
    def plot_top_attendees(summary_df: pd.DataFrame, top_n: int = 10) -> Figure:
        """
        Bar chart of the top N students by attendance percentage.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Output of AttendanceManager.calculate_attendance_percentage()
        top_n : int
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        if summary_df is None or summary_df.empty:
            ax.text(0.5, 0.5, "No data.", ha="center", va="center",
                    fontsize=14)
            ax.set_axis_off()
            return fig

        top = summary_df.nlargest(top_n, "Attendance_Percentage")

        bars = ax.bar(
            top["Student_ID"],
            top["Attendance_Percentage"],
            color=COLORS[2],
            alpha=0.85,
            edgecolor=BG_COLOR,
        )

        for bar, row in zip(bars, top.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{row.Attendance_Percentage:.1f}%",
                ha="center", fontsize=9
            )

        ax.axhline(y=75, color=COLORS[3], linestyle="--",
                   linewidth=1.5, label="75% threshold")
        ax.set_ylim(0, 115)
        ax.set_xlabel("Student ID",               fontsize=12)
        ax.set_ylabel("Attendance Percentage (%)", fontsize=12)
        ax.set_title(f"Top {top_n} Students by Attendance",
                     fontsize=14, fontweight="bold", pad=15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        return fig