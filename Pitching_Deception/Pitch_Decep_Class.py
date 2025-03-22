import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from pybaseball import statcast_pitcher, pitching_stats, pitching_stats_bref, playerid_lookup
from sklearn.cluster import KMeans
import os

class Pitching_Deception_Calc:
    def __init__(self, csv_folder='CSV_Files'):
        self.csv_folder = csv_folder
        self.plots_folder = 'Results'
        os.makedirs(self.plots_folder, exist_ok=True)
        os.makedirs(self.csv_folder, exist_ok=True)

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def retrieve_and_save_data(self, player_name_list, start_date, end_date):
        for player in player_name_list:
            last_name, first_name = player.split()
            try:
                player_id = playerid_lookup(last_name, first_name)['key_mlbam'].iloc[0]
                data = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=player_id)
                if not data.empty:
                    filename = f"{last_name}_{first_name}_2024.csv"
                    filepath = os.path.join(self.csv_folder, filename)
                    data.to_csv(filepath, index=False)
                    print(f"Data saved to {filepath}")
                else:
                    print(f"No data available for {player}")
            except Exception as e:
                print(f"An error occurred while processing {player}: {str(e)}")

    def analyze_pitching_deception(self, data, league_average_smr=0.11):
        data = data.dropna(subset=['effective_speed', 'release_spin_rate', 'release_speed', 'player_name', 'p_throws'])
        data = data.dropna(axis=1)
        data = data.reset_index(drop=True)

        pitcher_name = data['player_name'].iloc[0]
        pitcher_hand = data['p_throws'].iloc[0]

        if pitcher_hand == 'L':
            data['release_pos_x'] = data['release_pos_x'] * -1

        pitch_counts = data.groupby('pitch_type').size()
        total_swings = data[data['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip', 'missed_bunt', 'hit_into_play'])].groupby('pitch_type').size()
        total_misses = data[data['description'] == 'swinging_strike'].groupby('pitch_type').size()

        swing_and_miss_rate = pd.DataFrame({
            'pitch_type': pitch_counts.index,
            'total_pitches': pitch_counts.values,
            'total_swings': total_swings.reindex(pitch_counts.index, fill_value=0).values,
            'total_misses': total_misses.reindex(pitch_counts.index, fill_value=0).values
        })

        swing_and_miss_rate['swing_and_miss_rate'] = np.where(
            swing_and_miss_rate['total_swings'] > 0,
            swing_and_miss_rate['total_misses'] / swing_and_miss_rate['total_swings'],
            0
        )

        swing_and_miss_rate['pitch_usage'] = swing_and_miss_rate['total_pitches'] / swing_and_miss_rate['total_pitches'].sum()
        swing_and_miss_rate['normalized_smr'] = swing_and_miss_rate['swing_and_miss_rate'] / league_average_smr * 100
        swing_and_miss_rate['weighted_normalized_smr'] = swing_and_miss_rate['normalized_smr'] * swing_and_miss_rate['pitch_usage']

        smr_score = swing_and_miss_rate['weighted_normalized_smr'].sum()

        pitch_summary = data.groupby('pitch_type').agg({
            'release_pos_x': 'mean',
            'release_pos_y': 'mean',
            'release_pos_z': 'mean',
            'effective_speed': 'mean',
            'release_speed': 'mean'
        }).reset_index()

        pitch_summary = pitch_summary.merge(swing_and_miss_rate, on='pitch_type', how='left')
        pitch_summary = pitch_summary.sort_values('swing_and_miss_rate')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(pitch_summary)))

        scatter = ax1.scatter(pitch_summary['release_pos_x'],
                              pitch_summary['release_pos_z'],
                              s=pitch_summary['effective_speed'] * 2,
                              c=colors,
                              alpha=0.7)

        for idx, row in pitch_summary.iterrows():
            ax1.annotate(f"{row['pitch_type']}\n({row['release_pos_x']:.2f}, {row['release_pos_z']:.2f})",
                         (row['release_pos_x'], row['release_pos_z']),
                         xytext=(10, 5),
                         textcoords='offset points',
                         fontsize=8,
                         alpha=0.8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax1.set_title(f'Spray Chart of Release Position for {pitcher_name}')
        ax1.set_xlabel('Release Position X')
        ax1.set_ylabel('Release Position Z')

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      label=pitch, markerfacecolor=color, markersize=10)
                           for pitch, color in zip(pitch_summary['pitch_type'], colors)]
        ax1.legend(handles=legend_elements, title='Pitch Types', loc='best')

        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.barh(pitch_summary['pitch_type'], pitch_summary['swing_and_miss_rate'], color=colors)
        ax2.set_title(f'Swing-and-Miss Rates for {pitcher_name}')
        ax2.set_xlabel('Rate')
        ax2.set_xlim(0, 1)

        for i, v in enumerate(pitch_summary['swing_and_miss_rate']):
            ax2.text(v, i, f' {v:.1%}', va='center')

        ax2.grid(True, linestyle='--', alpha=0.7)

        ff_row = pitch_summary[pitch_summary['pitch_type'] == 'FF']
        ch_row = pitch_summary[pitch_summary['pitch_type'] == 'CH']

        if not ff_row.empty and not ch_row.empty:
            ff_position = (ff_row['release_pos_x'].values[0], ff_row['release_pos_z'].values[0])
            ch_position = (ch_row['release_pos_x'].values[0], ch_row['release_pos_z'].values[0])

            ax1.plot([ff_position[0], ch_position[0]],
                     [ff_position[1], ch_position[1]],
                     color='red',
                     linewidth=2)

            distance = np.sqrt((ff_position[0] - ch_position[0]) ** 2 + (ff_position[1] - ch_position[1]) ** 2)
            mid_point_x = (ff_position[0] + ch_position[0]) / 2
            mid_point_y = (ff_position[1] + ch_position[1]) / 2

            ax1.annotate(f'Distance: {distance:.2f} ft',
                         xy=(mid_point_x, mid_point_y),
                         xytext=(mid_point_x + 5, mid_point_y + 5),
                         fontsize=10,
                         color='red',
                         arrowprops=dict(facecolor='red', shrink=0.05))

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, f'{pitcher_name}_Pitch_Analysis.png'))
        plt.close()

        return smr_score, swing_and_miss_rate

    def calculate_wcr(self, data, launch_angle_thresholds=(10, 25), exit_velocity_threshold=80):
        weak_contact_condition = (
            (data['launch_angle'] >= launch_angle_thresholds[0]) &
            (data['launch_angle'] <= launch_angle_thresholds[1]) &
            (data['launch_speed'] < exit_velocity_threshold)
        )
        
        total_batted_balls = data.shape[0]
        weak_contact_count = weak_contact_condition.sum()
        
        wcr = (weak_contact_count / total_batted_balls) * 100 if total_batted_balls > 0 else 0
        
        return wcr

    def analyze_pitcher_deception(self):
        plots_folder = 'Launch_Angle_Flyout_Plots'
        os.makedirs(plots_folder, exist_ok=True)
        
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            player_name = ' '.join(csv_file.split('_')[:2])
            file_path = os.path.join(self.csv_folder, csv_file)
            data = pd.read_csv(file_path)
            
            pitcher_data = data.dropna(subset=['events', 'description', 'launch_angle', 'launch_speed'])
            
            flyout_condition = (
                ((pitcher_data['events'] == 'field_out') & (pitcher_data['description'].isin(['hit_into_play', 'foul']))) |
                (pitcher_data['events'] == 'sac_fly')
            )
            
            pitcher_data['is_flyout'] = flyout_condition
            
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=pitcher_data, x='launch_speed', y='launch_angle', hue='is_flyout', palette={True: 'red', False: 'blue'})
            
            plt.axhline(y=25, color='g', linestyle='--', label='Fly ball lower limit')
            plt.axhline(y=50, color='g', linestyle='--', label='Fly ball upper limit')
            
            plt.title(f'Launch Angle vs. Exit Velocity for {player_name}')
            plt.xlabel('Exit Velocity (mph)')
            plt.ylabel('Launch Angle (degrees)')
            plt.legend(title='Flyout', labels=['Other', 'Flyout'], bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_folder, f'{player_name}_Launch_Angle_vs_Exit_Velocity.png'))
            plt.close()
            
            total_batted_balls = pitcher_data.shape[0]
            fly_outs = pitcher_data['is_flyout'].sum()
            
            fly_out_percentage = (fly_outs / total_batted_balls) * 100 if total_batted_balls > 0 else 0
            
            print(f"Analyzing {player_name}:")
            print(f"Fly out percentage: {fly_out_percentage:.2f}%")
            
            wcr_score = self.calculate_wcr(pitcher_data)
            
            print(f"Weak Contact Rate (WCR) Score for {player_name}: {wcr_score:.2f}%")
            
            pitch_type_analysis = pitcher_data.groupby('pitch_type').agg({
                'launch_angle': 'mean',
                'launch_speed': 'mean',
                'is_flyout': lambda x: x.sum() / len(x) * 100
            }).reset_index()

            ci_low, ci_high = zip(*pitcher_data.groupby('pitch_type')['is_flyout'].apply(self.bootstrap_flyout_percentage))
            pitch_type_analysis['CI_low'] = ci_low
            pitch_type_analysis['CI_high'] = ci_high

            pitch_type_analysis.columns = ['Pitch Type', 'Avg Launch Angle', 'Avg Exit Velocity', 'Fly Out %', 'CI_low', 'CI_high']
            
            print("\nPitch Type Analysis:")
            print(pitch_type_analysis)

            lower_error = pitch_type_analysis['Fly Out %'] - pitch_type_analysis['CI_low']
            upper_error = pitch_type_analysis['CI_high'] - pitch_type_analysis['Fly Out %']

            lower_error = np.maximum(lower_error, 0)
            upper_error = np.maximum(upper_error, 0)

            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=pitch_type_analysis, x='Pitch Type', y='Fly Out %')

            ax.errorbar(x=range(len(pitch_type_analysis)),
                        y=pitch_type_analysis['Fly Out %'],
                        yerr=[lower_error, upper_error],
                        fmt='none', color='black', capsize=5)

            plt.title(f'Fly Out Percentage by Pitch Type for {player_name}')
            plt.ylabel('Fly Out Percentage')
            plt.xticks(rotation=45)

            for i, v in enumerate(pitch_type_analysis['Fly Out %']):
                ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, f'{player_name}_Fly_Out_Percentage_by_Pitch_Type.png'))
            plt.close()

            print("\nFly Out Percentages with 95% Confidence Intervals:")
            for _, row in pitch_type_analysis.iterrows():
                print(f"{row['Pitch Type']}: {row['Fly Out %']:.2f}% (95% CI: {row['CI_low']:.2f}% - {row['CI_high']:.2f}%)")

    """Weak Contact Rate (WCR) caluclation. Fly out percentage/Plots Section"""
    def bootstrap_flyout_percentage(self, data, num_bootstrap_samples=1000):
        bootstrap_means = []
        for _ in range(num_bootstrap_samples):
            sample = data.sample(n=len(data), replace=True)
            bootstrap_means.append(sample.mean())
        return np.percentile(bootstrap_means, [2.5, 97.5])

    def calculate_wcr(self, data, launch_angle_thresholds=(10, 25), exit_velocity_threshold=80):
        weak_contact_condition = (
            (data['launch_angle'] >= launch_angle_thresholds[0]) &
            (data['launch_angle'] <= launch_angle_thresholds[1]) &
            (data['launch_speed'] < exit_velocity_threshold)
        )
        
        total_batted_balls = data.shape[0]
        weak_contact_count = weak_contact_condition.sum()
        
        wcr = (weak_contact_count / total_batted_balls) * 100 if total_batted_balls > 0 else 0
        
        return wcr

    def launch_angle_plots(self, data, player_name):
        plots_folder = os.path.join(self.plots_folder, 'Launch_Angle_Flyout_Plots')
        os.makedirs(plots_folder, exist_ok=True)
        
        pitcher_data = data.dropna(subset=['events', 'description', 'launch_angle', 'launch_speed'])
        
        flyout_condition = (
            ((pitcher_data['events'] == 'field_out') & (pitcher_data['description'].isin(['hit_into_play', 'foul']))) |
            (pitcher_data['events'] == 'sac_fly')
        )
        
        pitcher_data['is_flyout'] = flyout_condition
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=pitcher_data, x='launch_speed', y='launch_angle', hue='is_flyout', palette={True: 'red', False: 'blue'})
        
        plt.axhline(y=25, color='g', linestyle='--', label='Fly ball lower limit')
        plt.axhline(y=50, color='g', linestyle='--', label='Fly ball upper limit')
        
        plt.title(f'Launch Angle vs. Exit Velocity for {player_name}')
        plt.xlabel('Exit Velocity (mph)')
        plt.ylabel('Launch Angle (degrees)')
        plt.legend(title='Flyout', labels=['Other', 'Flyout'], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_folder, f'{player_name}_Launch_Angle_vs_Exit_Velocity.png'))
        plt.close()
        
        total_batted_balls = pitcher_data.shape[0]
        fly_outs = pitcher_data['is_flyout'].sum()
        
        fly_out_percentage = (fly_outs / total_batted_balls) * 100 if total_batted_balls > 0 else 0
        
        wcr_score = self.calculate_wcr(pitcher_data)
        
        pitch_type_analysis = pitcher_data.groupby('pitch_type').agg({
            'launch_angle': 'mean',
            'launch_speed': 'mean',
            'is_flyout': lambda x: x.sum() / len(x) * 100
        }).reset_index()

        ci_low, ci_high = zip(*pitcher_data.groupby('pitch_type')['is_flyout'].apply(self.bootstrap_flyout_percentage))
        pitch_type_analysis['CI_low'] = ci_low
        pitch_type_analysis['CI_high'] = ci_high

        pitch_type_analysis.columns = ['Pitch Type', 'Avg Launch Angle', 'Avg Exit Velocity', 'Fly Out %', 'CI_low', 'CI_high']
        
        lower_error = pitch_type_analysis['Fly Out %'] - pitch_type_analysis['CI_low']
        upper_error = pitch_type_analysis['CI_high'] - pitch_type_analysis['Fly Out %']

        lower_error = np.maximum(lower_error, 0)
        upper_error = np.maximum(upper_error, 0)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=pitch_type_analysis, x='Pitch Type', y='Fly Out %')

        ax.errorbar(x=range(len(pitch_type_analysis)), 
                    y=pitch_type_analysis['Fly Out %'],
                    yerr=[lower_error, upper_error],
                    fmt='none', color='black', capsize=5)

        plt.title(f'Fly Out Percentage by Pitch Type for {player_name}')
        plt.ylabel('Fly Out Percentage')
        plt.xticks(rotation=45)

        for i, v in enumerate(pitch_type_analysis['Fly Out %']):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_folder, f'{player_name}_Fly_Out_Percentage_by_Pitch_Type.png'))
        plt.close()

        return fly_out_percentage, wcr_score, pitch_type_analysis
    
    """Pitch Movmement Plots"""

    def create_movement_plot(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        plot_data = data.dropna(subset=['player_name', 'pitch_type', 'pfx_x', 'pfx_z', 'release_spin_rate'])

        pitcher_name = plot_data['player_name'].iloc[0]

        avg_spin_rates = plot_data.groupby('pitch_type')['release_spin_rate'].mean().round().astype(int)

        plt.figure(figsize=(12, 8))
        
        sns.scatterplot(data=plot_data, x='pfx_x', y='pfx_z',
                        hue='pitch_type', palette='deep')

        plt.title(f'Pitch Movement Plot for {pitcher_name}', fontsize=16)
        plt.xlabel('Horizontal Movement (inches)', fontsize=12)
        plt.ylabel('Vertical Movement (inches)', fontsize=12)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

        for pitch_type in avg_spin_rates.index:
            pitch_data = plot_data[plot_data['pitch_type'] == pitch_type]
            avg_x = pitch_data['pfx_x'].mean()
            avg_y = pitch_data['pfx_z'].mean()
            plt.annotate(f"{pitch_type}\n{avg_spin_rates[pitch_type]} rpm",
                         (avg_x, avg_y),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center',
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, f'{pitcher_name}_Pitch_Movement.png'))
        plt.close()

    def calculate_expected_movement(self, velocity, spin_rate, pitch_type):
        k_horizontal = 0.00006
        k_vertical = 0.00004
        
        expected_horizontal = k_horizontal * spin_rate / velocity
        expected_vertical = k_vertical * spin_rate / velocity
        
        if pitch_type in ['FF', 'FT']:
            expected_vertical *= 1.2
        elif pitch_type in ['SL', 'CU']:
            expected_horizontal *= 1.5
            expected_vertical *= -1
        elif pitch_type == 'CH':
            expected_vertical *= 0.8
        
        return expected_horizontal, expected_vertical

    def calculate_pmd_score(self, data, league_average_deviation=2.0):
        data['total_deviation'] = np.sqrt(data['x_deviation']**2 + data['z_deviation']**2)
        
        pitch_deviations = data.groupby('pitch_type')['total_deviation'].mean()
        
        pitch_usage = data['pitch_type'].value_counts(normalize=True)
        
        normalized_deviations = pitch_deviations / league_average_deviation
        
        pmd_score = (normalized_deviations * pitch_usage).sum() * 100
        
        return pmd_score, pitch_deviations

    def analyze_pitch_deception(self, data):
        data['expected_pfx_x'], data['expected_pfx_z'] = zip(*data.apply(
            lambda row: self.calculate_expected_movement(row['release_speed'], row['release_spin_rate'], row['pitch_type']),
            axis=1
        ))
        
        data['x_deviation'] = data['pfx_x'] - data['expected_pfx_x']
        data['z_deviation'] = data['pfx_z'] - data['expected_pfx_z']
        
        pmd_score, pitch_deviations = self.calculate_pmd_score(data)
        
        return data, pmd_score, pitch_deviations

    def plot_movement_comparison(self, data, pitcher_name):
        pitch_types = data['pitch_type'].unique()
        n_pitches = len(pitch_types)
        
        n_cols = 3
        n_rows = (n_pitches - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
        fig.suptitle(f'Actual vs Expected Movement - {pitcher_name}', fontsize=16)
        
        for idx, pitch_type in enumerate(pitch_types):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            pitch_data = data[data['pitch_type'] == pitch_type]
            
            ax.scatter(pitch_data['pfx_x'], pitch_data['pfx_z'],
                       s=50, alpha=0.7, color='blue', label='Actual')
            
            ax.scatter(pitch_data['expected_pfx_x'], pitch_data['expected_pfx_z'],
                       color='red', marker='x', s=50, alpha=0.7, label='Expected')
            
            for _, pitch in pitch_data.iterrows():
                ax.plot([pitch['pfx_x'], pitch['expected_pfx_x']],
                        [pitch['pfx_z'], pitch['expected_pfx_z']],
                        color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_title(f'{pitch_type}')
            ax.set_xlabel('Horizontal Movement (inches)')
            ax.set_ylabel('Vertical Movement (inches)')
            
            ax.legend()
        
        for idx in range(n_pitches, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, f'{pitcher_name}_Actual_vs_Expected_Movement.png'))
        plt.close()

    def analyze_release_point_consistency(self, data, pitcher_name):
        clean_data = data.dropna(subset=['pitch_type', 'release_pos_x', 'release_pos_y', 'release_pos_z'])

        for col in ['release_pos_x', 'release_pos_y', 'release_pos_z']:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]

        def calculate_distance_metrics(group):
            release_points = group[['release_pos_x', 'release_pos_y', 'release_pos_z']].values
            if len(release_points) < 2:
                return pd.Series({
                    'mean_distance': np.nan,
                    'std_distance_x': np.nan,
                    'std_distance_y': np.nan,
                    'std_distance_z': np.nan,
                    'count': len(group)
                })
            distances = pdist(release_points)
            return pd.Series({
                'mean_distance': np.mean(distances),
                'std_distance_x': np.std(group['release_pos_x']),
                'std_distance_y': np.std(group['release_pos_y']),
                'std_distance_z': np.std(group['release_pos_z']),
                'count': len(group)
            })

        consistency_metrics = clean_data.groupby('pitch_type').apply(calculate_distance_metrics)
        
        consistency_metrics = consistency_metrics.sort_values('mean_distance', na_position='last')
        
        plt.figure(figsize=(12, 6))
        
        bar_plot = sns.barplot(x=consistency_metrics.index,
                               y='mean_distance',
                               data=consistency_metrics,
                               palette='Blues',
                               ci=None)

        x_positions = np.arange(len(consistency_metrics.index))
        
        offset_width = 0.15

        plt.errorbar(x=x_positions - offset_width,
                     y=consistency_metrics['mean_distance'],
                     yerr=consistency_metrics['std_distance_x'],
                     fmt='none',
                     c='red',
                     capsize=5,
                     label='Std Dev X')

        plt.errorbar(x=x_positions + offset_width,
                     y=consistency_metrics['mean_distance'],
                     yerr=consistency_metrics['std_distance_y'],
                     fmt='none',
                     c='green',
                     capsize=5,
                     label='Std Dev Y')

        plt.errorbar(x=x_positions + offset_width * 2,
                     y=consistency_metrics['mean_distance'],
                     yerr=consistency_metrics['std_distance_z'],
                     fmt='none',
                     c='blue',
                     capsize=5,
                     label='Std Dev Z')

        for i, row in enumerate(consistency_metrics.itertuples()):
            plt.text(i, row.mean_distance + 0.5, f'n={row.count}', ha='center', va='bottom')

        plt.legend(title='Standard Deviations')
        
        plt.title(f'Release Point Consistency by Pitch Type - {pitcher_name}')
        plt.xlabel('Pitch Type')
        
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_folder, f'{pitcher_name}_Release_Point_Consistency.png'))
        plt.close()



    def calculate_rpd_score(self, data, league_average_std=0.1):
        release_std = data.groupby('pitch_type').agg({
            'release_pos_x': 'std',
            'release_pos_z': 'std'
        }).mean(axis=1)
        
        pitch_usage = data['pitch_type'].value_counts(normalize=True)
        
        normalized_rpd = league_average_std / release_std
        
        rpd_score = (normalized_rpd * pitch_usage).sum() * 100
        
        return rpd_score

    def create_movement_release_plots(self, data, pitcher_name, pitch_selection='all'):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        plot_data = data.dropna(subset=['pitch_type', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z', 'release_spin_rate', 'p_throws'])

        plot_data = plot_data.copy()
        if plot_data['p_throws'].iloc[0] == 'L':
            plot_data['release_pos_x'] = plot_data['release_pos_x'] * -1

        if pitch_selection == 'FF_CH':
            selected_pitches = ['FF', 'CH']
            pitch_types = [pt for pt in selected_pitches if pt in plot_data['pitch_type'].unique()]
            if len(pitch_types) == 0:
                raise ValueError("Neither 'FF' nor 'CH' found in the dataset")
        else:
            pitch_types = plot_data['pitch_type'].unique()

        rpd_score = self.calculate_rpd_score(plot_data)

        plots_folder = os.path.join(self.plots_folder, 'Movement_Release_Plots')
        os.makedirs(plots_folder, exist_ok=True)

        for pitch_type in pitch_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Movement vs Release Position for {pitch_type} - {pitcher_name}', fontsize=16)
            
            pitch_data = plot_data[plot_data['pitch_type'] == pitch_type]
            
            sns.scatterplot(data=pitch_data, x='release_pos_x', y='pfx_x', ax=ax1)
            ax1.set_title('Horizontal Movement', fontsize=14)
            ax1.set_xlabel('Release Position X (ft)', fontsize=12)
            ax1.set_ylabel('Horizontal Movement (inches)', fontsize=12)
            ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

            sns.scatterplot(data=pitch_data, x='release_pos_z', y='pfx_z', ax=ax2)
            ax2.set_title('Vertical Movement', fontsize=14)
            ax2.set_xlabel('Release Position Z (ft)', fontsize=12)
            ax2.set_ylabel('Vertical Movement (inches)', fontsize=12)
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

            avg_spin_rate = pitch_data['release_spin_rate'].mean().round().astype(int)
            fig.text(0.5, 0.02, f'Average Spin Rate: {avg_spin_rate} rpm', ha='center', fontsize=12)

            handedness = "Left-handed" if plot_data['p_throws'].iloc[0] == 'L' else "Right-handed"
            fig.text(0.5, 0.97, f'{handedness} Pitcher', ha='center', fontsize=12)

            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_folder, f'{pitcher_name}_{pitch_type}_Movement_Plot.png'))
            
            plt.close(fig)

        return rpd_score

    def run_analysis(self):
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        rpd_scores = {}
        wcr_scores = {}

        for csv_file in csv_files:
            pitcher_name = '_'.join(csv_file.split('_')[:2])
            file_path = os.path.join(self.csv_folder, csv_file)
            try:
                data = self.load_data(file_path)
                print(f"Processing {pitcher_name}...")
                
                smr_score, swing_and_miss_rate = self.analyze_pitching_deception(data)
                print(f"SMR Score for {pitcher_name}: {smr_score:.2f}")
                
                self.create_movement_plot(data)
                
                analyzed_data, pmd_score, pitch_deviations = self.analyze_pitch_deception(data)
                print(f"PMD Score for {pitcher_name}: {pmd_score:.2f}")
                
                self.plot_movement_comparison(analyzed_data, pitcher_name)
                
                self.analyze_release_point_consistency(data, pitcher_name)
                
                rpd_score = self.create_movement_release_plots(data, pitcher_name, pitch_selection='FF_CH')
                rpd_scores[pitcher_name] = rpd_score
                print(f"RPD Score for {pitcher_name}: {rpd_score:.2f}")
                
                fly_out_percentage, wcr_score, pitch_type_analysis = self.launch_angle_plots(data, pitcher_name)
                wcr_scores[pitcher_name] = wcr_score
                print(f"Fly out percentage: {fly_out_percentage:.2f}%")
                print(f"Weak Contact Rate (WCR) Score for {pitcher_name}: {wcr_score:.2f}%")
                
                print("\nPitch Type Analysis:")
                print(pitch_type_analysis)
                
                print("\nFly Out Percentages with 95% Confidence Intervals:")
                for _, row in pitch_type_analysis.iterrows():
                    print(f"{row['Pitch Type']}: {row['Fly Out %']:.2f}% (95% CI: {row['CI_low']:.2f}% - {row['CI_high']:.2f}%)")
                
                print(f"Analysis complete for {pitcher_name}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {pitcher_name}: {e}")
                with open('error_log.txt', 'a') as f:
                    f.write(f"Error occurred at {pd.Timestamp.now()}: {str(e)}\n")
