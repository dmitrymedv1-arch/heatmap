import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import io
import base64
import colorcet as cc
import warnings
warnings.filterwarnings('ignore')

# Импорт статистических библиотек
from scipy import stats
from scipy.stats import zscore, shapiro, normaltest, ttest_ind, ttest_rel, f_oneway, levene, mannwhitneyu, wilcoxon
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import pingouin as pg

# Настройка стиля для научной статьи
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['text.color'] = 'black'

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# ============================================================================

class WarningSystem:
    """Warning and recommendation system"""
    
    def __init__(self):
        self.warnings = []
        self.recommendations = []
    
    def check_imputation_danger(self, strategy, data_type, missing_percent):
        """Check imputation danger"""
        if strategy == 'zero' and missing_percent > 5:
            self.add_warning(
                "⚠️ ZERO IMPUTATION",
                f"Imputing {missing_percent:.1f}% missing values with zeros may significantly distort statistics!",
                "Consider using mean/median per row or KNN imputation"
            )
        elif strategy == 'drop' and missing_percent > 30:
            self.add_warning(
                "⚠️ DATA DELETION",
                f"Deleting {missing_percent:.1f}% of data may lead to loss of information",
                "Consider imputation instead of deletion"
            )
    
    def check_statistical_power(self, n_samples, n_tests, test_type):
        """Check statistical power"""
        if n_samples < 5:
            self.add_warning(
                "⚠️ SMALL SAMPLE SIZE",
                f"Only {n_samples} samples for {n_tests} tests",
                "Results may be unreliable. Consider non-parametric tests"
            )
        elif n_tests > 50 and n_samples < 10:
            self.add_warning(
                "⚠️ MULTIPLE TESTING",
                f"{n_tests} tests with {n_samples} samples",
                "Multiple testing correction may be too conservative"
            )
    
    def check_normalization(self, data, method):
        """Check normalization"""
        if method == 'log_transform' and np.any(data <= 0):
            self.add_warning(
                "⚠️ LOG TRANSFORMATION",
                "Negative or zero values in data",
                "Using log1p (log(1+x)), but this may not be optimal"
            )
    
    def add_warning(self, title, message, recommendation):
        self.warnings.append(f"{title}: {message}")
        self.recommendations.append(recommendation)
    
    def display_warnings(self):
        """Display all warnings"""
        if self.warnings:
            st.warning("Warnings and Recommendations")
            for i, warning in enumerate(self.warnings, 1):
                st.write(f"{i}. {warning}")
            if self.recommendations:
                st.info("Recommendations:")
                for i, rec in enumerate(self.recommendations, 1):
                    st.write(f"{i}. {rec}")

class MissingValueHandler:
    """Missing value handler"""
    
    def __init__(self):
        self.strategies = {
            'none': self.no_imputation,
            'zero': self.impute_zero,
            'mean_row': self.impute_row_mean,
            'mean_col': self.impute_col_mean,
            'mean_all': self.impute_mean_all,
            'median_row': self.impute_row_median,
            'median_col': self.impute_col_median,
            'knn': self.impute_knn,
            'drop_rows': self.drop_rows_with_nan,
            'drop_cols': self.drop_cols_with_nan
        }
        
        self.warning_system = WarningSystem()
    
    def handle(self, data, strategy='mean_row', **kwargs):
        """Handle missing values with selected strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Check for missing values
        missing_percent = (np.isnan(data).sum() / data.size) * 100
        
        if missing_percent > 0:
            self.warning_system.check_imputation_danger(strategy, 'generic', missing_percent)
            st.info(f"Found missing values: {missing_percent:.1f}%")
        
        # Apply strategy
        result = self.strategies[strategy](data.copy(), **kwargs)
        
        # Check result
        if np.any(np.isnan(result)):
            st.warning("Remaining missing values after imputation. Replacing with 0.")
            result = np.nan_to_num(result)
        
        return result
    
    def no_imputation(self, data, **kwargs):
        return data
    
    def impute_zero(self, data, **kwargs):
        return np.nan_to_num(data)
    
    def impute_row_mean(self, data, **kwargs):
        row_means = np.nanmean(data, axis=1, keepdims=True)
        row_indices, col_indices = np.where(np.isnan(data))
        data[row_indices, col_indices] = row_means[row_indices, 0]
        return data
    
    def impute_col_mean(self, data, **kwargs):
        col_means = np.nanmean(data, axis=0, keepdims=True)
        row_indices, col_indices = np.where(np.isnan(data))
        data[row_indices, col_indices] = col_means[0, col_indices]
        return data
    
    def impute_mean_all(self, data, **kwargs):
        global_mean = np.nanmean(data)
        data[np.isnan(data)] = global_mean
        return data
    
    def impute_row_median(self, data, **kwargs):
        row_medians = np.nanmedian(data, axis=1, keepdims=True)
        row_indices, col_indices = np.where(np.isnan(data))
        data[row_indices, col_indices] = row_medians[row_indices, 0]
        return data
    
    def impute_col_median(self, data, **kwargs):
        col_medians = np.nanmedian(data, axis=0, keepdims=True)
        row_indices, col_indices = np.where(np.isnan(data))
        data[row_indices, col_indices] = col_medians[0, col_indices]
        return data
    
    def impute_knn(self, data, n_neighbors=5, **kwargs):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return imputer.fit_transform(data)
    
    def drop_rows_with_nan(self, data, threshold=0.5, **kwargs):
        # Remove rows with more than threshold% missing values
        nan_per_row = np.isnan(data).mean(axis=1)
        keep_rows = nan_per_row < threshold
        return data[keep_rows, :]
    
    def drop_cols_with_nan(self, data, threshold=0.5, **kwargs):
        # Remove columns with more than threshold% missing values
        nan_per_col = np.isnan(data).mean(axis=0)
        keep_cols = nan_per_col < threshold
        return data[:, keep_cols]

class NormalizationEngine:
    """Data normalization engine"""
    
    def __init__(self):
        self.methods = {
            'none': lambda x: x,
            'zscore_rows': self.zscore_rows,
            'zscore_cols': self.zscore_cols,
            'zscore_all': self.zscore_all,
            'minmax_rows': self.minmax_rows,
            'minmax_cols': self.minmax_cols,
            'minmax_all': self.minmax_all,
            'robust_rows': self.robust_rows,
            'robust_cols': self.robust_cols,
            'log_transform': self.log_transform,
            'log2_transform': self.log2_transform,
            'quantile': self.quantile_normalize
        }
    
    def normalize(self, data, method='none'):
        """Apply normalization"""
        if method not in self.methods:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return self.methods[method](data.copy())
    
    def zscore_rows(self, data):
        return zscore(data, axis=1, nan_policy='omit')
    
    def zscore_cols(self, data):
        return zscore(data, axis=0, nan_policy='omit')
    
    def zscore_all(self, data):
        return zscore(data.flatten(), nan_policy='omit').reshape(data.shape)
    
    def minmax_rows(self, data):
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            row = data[i, :]
            min_val = np.nanmin(row)
            max_val = np.nanmax(row)
            if max_val > min_val:
                result[i, :] = (row - min_val) / (max_val - min_val)
        return result
    
    def minmax_cols(self, data):
        result = np.zeros_like(data)
        for j in range(data.shape[1]):
            col = data[:, j]
            min_val = np.nanmin(col)
            max_val = np.nanmax(col)
            if max_val > min_val:
                result[:, j] = (col - min_val) / (max_val - min_val)
        return result
    
    def minmax_all(self, data):
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return data
    
    def robust_rows(self, data):
        scaler = RobustScaler()
        return np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() 
                        for row in data])
    
    def robust_cols(self, data):
        scaler = RobustScaler()
        return scaler.fit_transform(data.T).T
    
    def log_transform(self, data):
        # Use log1p to handle zeros and negative values
        data_min = np.nanmin(data)
        if data_min <= 0:
            shift = abs(data_min) + 1
            return np.log(data + shift)
        return np.log(data)
    
    def log2_transform(self, data):
        data_min = np.nanmin(data)
        if data_min <= 0:
            shift = abs(data_min) + 1
            return np.log2(data + shift)
        return np.log2(data)
    
    def quantile_normalize(self, data):
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        return qt.fit_transform(data)

class ClusteringManager:
    """Clustering manager with intelligent sorting"""
    
    def __init__(self):
        self.row_order = None
        self.col_order = None
        self.row_clusters = None
        self.col_clusters = None
    
    def cluster_data(self, data, rows=True, cols=True, method='hierarchical', 
                    n_clusters=3, metric='euclidean', linkage_method='ward'):
        """Cluster data while preserving order"""
        
        if rows:
            self.row_order, self.row_clusters = self._cluster_axis(
                data, axis=0, method=method, n_clusters=n_clusters,
                metric=metric, linkage_method=linkage_method
            )
        
        if cols:
            self.col_order, self.col_clusters = self._cluster_axis(
                data.T, axis=0, method=method, n_clusters=n_clusters,
                metric=metric, linkage_method=linkage_method
            )
    
    def _cluster_axis(self, data, axis=0, method='hierarchical', n_clusters=3,
                     metric='euclidean', linkage_method='ward'):
        """Cluster along one axis"""
        
        if method == 'hierarchical':
            # Hierarchical clustering
            if metric == 'correlation':
                dist_matrix = 1 - np.corrcoef(data)
            else:
                dist_matrix = pdist(data, metric=metric)
            
            Z = linkage(dist_matrix, method=linkage_method)
            
            if n_clusters > 1:
                clusters = fcluster(Z, n_clusters, criterion='maxclust')
                # Get leaf order from dendrogram
                leaves = leaves_list(Z)
                # Sort within clusters
                order = self._sort_within_clusters(leaves, clusters, data)
            else:
                order = leaves_list(Z)
                clusters = np.ones(len(data), dtype=int)
        
        elif method == 'kmeans':
            # KMeans with intelligent sorting
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data)
            order = self._sort_kmeans_clusters(data, clusters, kmeans.cluster_centers_)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return order, clusters
    
    def _sort_within_clusters(self, leaves, clusters, data):
        """Sort within clusters"""
        order = []
        
        # For each cluster
        for cluster_id in np.unique(clusters):
            # Indices of elements in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            if len(cluster_indices) > 1:
                # Sort by mean value
                cluster_data = data[cluster_indices]
                cluster_means = cluster_data.mean(axis=1)
                sorted_indices = np.argsort(cluster_means)
                order.extend([cluster_indices[i] for i in sorted_indices])
            else:
                order.extend(cluster_indices)
        
        return np.array(order)
    
    def _sort_kmeans_clusters(self, data, clusters, centers):
        """Intelligent sorting for KMeans"""
        # Sort clusters by their centers
        center_means = centers.mean(axis=1)
        cluster_order = np.argsort(center_means)
        
        order = []
        
        # For each cluster in sorted order
        for cluster_id in cluster_order:
            # Indices of elements in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 1:
                # Sort within cluster by distance to center
                cluster_data = data[cluster_indices]
                distances = np.linalg.norm(cluster_data - centers[cluster_id], axis=1)
                sorted_indices = np.argsort(distances)
                order.extend(cluster_indices[sorted_indices])
            else:
                order.extend(cluster_indices)
        
        return np.array(order)
    
    def apply_clustering(self, data, labels=None, axis=0):
        """Apply saved clustering order"""
        if axis == 0 and self.row_order is not None:
            data = data[self.row_order, :]
            if labels is not None:
                labels = [labels[i] for i in self.row_order]
        elif axis == 1 and self.col_order is not None:
            data = data[:, self.col_order]
            if labels is not None:
                labels = [labels[i] for i in self.col_order]
        
        return data, labels
    
    def get_dendrogram(self, data, axis=0, **kwargs):
        """Get dendrogram for visualization"""
        if axis == 0:
            Z = linkage(data, method='ward')
            return dendrogram(Z, **kwargs)
        else:
            Z = linkage(data.T, method='ward')
            return dendrogram(Z, **kwargs)

class StatisticalTestSelector:
    """Select correct statistical test"""
    
    def __init__(self):
        self.warning_system = WarningSystem()
    
    def select_test(self, group1, group2, paired=False, equal_var='auto', normality_check=True):
        """Select appropriate statistical test"""
        
        # Remove NaN
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]
        
        n1, n2 = len(group1_clean), len(group2_clean)
        
        # Check minimum sample size
        if n1 < 3 or n2 < 3:
            self.warning_system.add_warning(
                "⚠️ SMALL SAMPLE",
                f"Sample sizes: {n1} and {n2}",
                "Results may be unreliable"
            )
            return None, None, None
        
        # Check normality if required
        normality_ok = True
        if normality_check and n1 >= 3 and n2 >= 3:
            normality_ok = self._check_normality(group1_clean, group2_clean)
        
        # Select test
        if paired:
            # Paired tests
            if normality_ok and n1 == n2:
                test_func = ttest_rel
                test_name = "Paired t-test (Student)"
            else:
                test_func = wilcoxon
                test_name = "Wilcoxon test (paired)"
        else:
            # Unpaired tests
            if normality_ok:
                # Check variance equality
                if equal_var == 'auto':
                    _, p_levene = levene(group1_clean, group2_clean)
                    equal_var_bool = p_levene > 0.05
                else:
                    equal_var_bool = equal_var
                
                test_func = lambda x, y: ttest_ind(x, y, equal_var=equal_var_bool)
                test_name = f"{'Student' if equal_var_bool else 'Welch'} t-test"
            else:
                test_func = mannwhitneyu
                test_name = "Mann-Whitney U test"
        
        # Execute test
        try:
            if paired and test_func == wilcoxon:
                stat, p_value = test_func(group1_clean, group2_clean)
            else:
                stat, p_value = test_func(group1_clean, group2_clean)
        except Exception as e:
            self.warning_system.add_warning(
                "⚠️ TEST ERROR",
                str(e),
                "Check input data"
            )
            return None, None, None
        
        return test_func, test_name, (stat, p_value)
    
    def _check_normality(self, group1, group2):
        """Check normality of distributions"""
        if len(group1) < 3 or len(group2) < 3:
            return False
        
        # Shapiro-Wilk test (for small samples)
        if len(group1) <= 5000:
            _, p1 = shapiro(group1)
        else:
            # For large samples use normality test
            _, p1 = normaltest(group1)
        
        if len(group2) <= 5000:
            _, p2 = shapiro(group2)
        else:
            _, p2 = normaltest(group2)
        
        # Normality if both p > 0.05
        return p1 > 0.05 and p2 > 0.05
    
    def apply_multiple_testing_correction(self, p_values, method='auto', n_samples=None):
        """Multiple testing correction"""
        n_tests = len(p_values)
        
        # Automatic method selection
        if method == 'auto':
            if n_tests <= 10:
                method = 'none'
                self.warning_system.add_warning(
                    "ℹ️ CORRECTION",
                    f"No correction applied for {n_tests} tests",
                    "For small number of tests correction may be excessive"
                )
            elif n_samples and n_tests > 3 * n_samples:
                method = 'fdr_by'  # More conservative
                self.warning_system.add_warning(
                    "⚠️ CONSERVATIVE CORRECTION",
                    f"{n_tests} tests with {n_samples} samples",
                    "Using conservative Benjamini-Yekutieli correction"
                )
            else:
                method = 'fdr_bh'  # Standard FDR
        
        if method == 'none':
            return p_values, np.array([False] * len(p_values)), method
        
        # Apply correction
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method)
        
        return pvals_corrected, rejected, method

class ColorManager:
    """Color management for heatmap"""
    
    def __init__(self):
        # Color schemes
        self.palette_options = [
            # Divergent
            'RdBu_r', 'Spectral', 'RdYlBu_r', 'PiYG_r', 'PRGn_r', 'BrBG_r',
            'coolwarm', 'bwr', 'seismic',
            
            # Sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
            
            # Colorblind friendly
            'viridis', 'plasma', 'cividis',
            
            # Special
            'RdYlGn_r', 'bone', 'hot', 'gist_earth', 'terrain',
            
            # CET
            'cet_fire', 'cet_rainbow', 'cet_diverging_bwr_55_98_c37'
        ]
        
        self.palette_names = [
            'Red-Blue', 'Spectral', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG',
            'Coolwarm', 'Blue-White-Red', 'Seismic',
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
            'Viridis (CB)', 'Plasma (CB)', 'Cividis (CB)',
            'Red-Yellow-Green', 'Bone', 'Hot', 'Earth', 'Terrain',
            'Fire (CET)', 'Rainbow (CET)', 'Diverging BWR'
        ]
        
        self.palette_dict = dict(zip(self.palette_names, self.palette_options))
    
    def get_colormap(self, palette_name, custom_colors=None):
        """Get color map"""
        if custom_colors and custom_colors.strip():
            colors = [c.strip() for c in custom_colors.split(',') if c.strip()]
            if len(colors) >= 2:
                return LinearSegmentedColormap.from_list('custom', colors)
        
        palette = self.palette_dict.get(palette_name, 'coolwarm')
        
        if palette.startswith('cet_'):
            try:
                cmap_array = getattr(cc, palette.replace('cet_', ''))
                return ListedColormap(cmap_array)
            except:
                pass
        
        return plt.cm.get_cmap(palette)
    
    def get_text_color(self, cell_value, cmap, vmin, vmax, mode='auto'):
        """Determine optimal text color"""
        if mode == 'black':
            return 'black'
        elif mode == 'white':
            return 'white'
        
        # Normalize value
        if vmax > vmin:
            norm_value = (cell_value - vmin) / (vmax - vmin)
            norm_value = max(0, min(1, norm_value))
        else:
            norm_value = 0.5
        
        # Get background color
        rgba = cmap(norm_value)
        
        # Calculate luminance (WCAG 2.0)
        r, g, b = rgba[0], rgba[1], rgba[2]
        
        # Convert sRGB to linear space
        r_lin = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g_lin = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b_lin = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        
        luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        
        # Dynamic threshold
        threshold = 0.179  # WCAG contrast 4.5:1
        
        return 'white' if luminance < threshold else 'black'
    
    def check_color_contrast(self, cmap, values, center_at_zero=False):
        """Check contrast for set of values"""
        if center_at_zero:
            vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        
        problems = []
        test_values = [vmin, vmax, (vmin + vmax) / 2, 0] if center_at_zero else [vmin, vmax, (vmin + vmax) / 2]
        
        for val in test_values:
            color = self.get_text_color(val, cmap, vmin, vmax)
            # Check if text would be lost
            rgba = cmap((val - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            
            if (color == 'white' and brightness > 0.7) or (color == 'black' and brightness < 0.3):
                problems.append(f"Value {val:.2f}: low contrast")
        
        return problems

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

class ScientificHeatmapApp:
    def __init__(self):
        # Data
        self.raw_data = None
        self.data_matrix = None
        self.data_matrix_original = None
        self.processed_data = None
        self.display_data = None
        
        # Managers
        self.missing_handler = MissingValueHandler()
        self.normalization_engine = NormalizationEngine()
        self.clustering_manager = ClusteringManager()
        self.stat_test_selector = StatisticalTestSelector()
        self.color_manager = ColorManager()
        self.warning_system = WarningSystem()
        
        # Results
        self.statistics_results = {}
        self.processing_history = []
        
        # Settings
        self.current_settings = {}
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_done' not in st.session_state:
            st.session_state.analysis_done = False
        if 'plot_created' not in st.session_state:
            st.session_state.plot_created = False
    
    def setup_sidebar(self):
        """Setup sidebar widgets"""
        st.sidebar.title("🔬 Scientific Heatmap Analyzer v2.0")
        st.sidebar.markdown("---")
        
        # Data input method
        data_input_method = st.sidebar.radio(
            "Data Input Method",
            ["📝 Text Input", "📁 File Upload", "🧪 Sample Data"]
        )
        
        if data_input_method == "📝 Text Input":
            self.setup_text_input()
        elif data_input_method == "📁 File Upload":
            self.setup_file_upload()
        else:
            self.load_sample_data()
        
        if st.session_state.data_loaded:
            st.sidebar.markdown("---")
            
            # Create tabs for different sections
            tabs = st.sidebar.tabs(["📊 Preprocessing", "🧪 Statistics", "🎨 Visualization"])
            
            with tabs[0]:
                self.setup_preprocessing_widgets()
            
            with tabs[1]:
                self.setup_statistics_widgets()
            
            with tabs[2]:
                self.setup_visualization_widgets()
            
            # Action buttons
            st.sidebar.markdown("---")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("🔬 Analyze", use_container_width=True):
                    self.perform_analysis()
            
            with col2:
                if st.button("📈 Plot", use_container_width=True):
                    self.create_visualization()
            
            if st.sidebar.button("🔄 Reset", type="secondary", use_container_width=True):
                self.reset_analysis()
    
    def setup_text_input(self):
        """Setup text input widget"""
        st.sidebar.subheader("📝 Enter Data")
        
        data_format = st.sidebar.selectbox(
            "Data Format",
            ["X,Y,Value", "X\tY\tValue", "X Y Value"]
        )
        
        data_input = st.sidebar.text_area(
            "Enter your data",
            height=200,
            help="Enter data in the selected format. Example:\nGene1,Sample1,10.5\nGene1,Sample2,12.3"
        )
        
        delimiter_map = {
            "X,Y,Value": ",",
            "X\tY\tValue": "\t",
            "X Y Value": " "
        }
        
        if st.sidebar.button("Load Data", use_container_width=True):
            if data_input.strip():
                delimiter = delimiter_map[data_format]
                if self.parse_data_text(data_input, delimiter):
                    st.session_state.data_loaded = True
                    st.rerun()
    
    def setup_file_upload(self):
        """Setup file upload widget"""
        st.sidebar.subheader("📁 Upload Data")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'txt', 'xlsx', 'xls', 'tsv'],
            help="Upload CSV, TSV, Excel, or text file"
        )
        
        if uploaded_file is not None:
            delimiter = st.sidebar.selectbox(
                "Delimiter",
                [',', ';', '\t', ' ', 'auto']
            )
            
            if st.sidebar.button("Load File", use_container_width=True):
                content = uploaded_file.getvalue().decode('utf-8')
                if delimiter == 'auto':
                    delimiter = self.detect_delimiter(content)
                
                if self.parse_data_text(content, delimiter):
                    st.session_state.data_loaded = True
                    st.rerun()
    
    def load_sample_data(self):
        """Load sample data"""
        st.sidebar.subheader("🧪 Sample Data")
        
        if st.sidebar.button("Load 5x5 Test Data", use_container_width=True):
            self.load_test_data()
            st.session_state.data_loaded = True
            st.rerun()
    
    def setup_preprocessing_widgets(self):
        """Setup preprocessing widgets"""
        st.sidebar.subheader("🛠️ Preprocessing")
        
        self.imputation_strategy = st.sidebar.selectbox(
            "Imputation Strategy",
            [
                ("No imputation", "none"),
                ("Zeros (caution!)", "zero"),
                ("Row mean", "mean_row"),
                ("Column mean", "mean_col"),
                ("Row median", "median_row"),
                ("Column median", "median_col"),
                ("KNN imputation", "knn"),
                ("Drop rows", "drop_rows"),
                ("Drop columns", "drop_cols")
            ],
            format_func=lambda x: x[0]
        )[1]
        
        self.normalization_method = st.sidebar.selectbox(
            "Normalization",
            [
                ("No normalization", "none"),
                ("Z-score rows", "zscore_rows"),
                ("Z-score columns", "zscore_cols"),
                ("Min-max rows", "minmax_rows"),
                ("Min-max columns", "minmax_cols"),
                ("Log transform", "log_transform"),
                ("Log2 transform", "log2_transform"),
                ("Quantile normalization", "quantile")
            ],
            format_func=lambda x: x[0]
        )[1]
    
    def setup_statistics_widgets(self):
        """Setup statistics widgets"""
        st.sidebar.subheader("🧪 Statistical Analysis")
        
        self.test_type = st.sidebar.selectbox(
            "Test Type",
            [
                ("Auto select", "auto"),
                ("Paired t-test", "paired"),
                ("Unpaired t-test", "unpaired"),
                ("Mann-Whitney", "mannwhitney"),
                ("Wilcoxon", "wilcoxon")
            ],
            format_func=lambda x: x[0]
        )[1]
        
        self.equal_variance = st.sidebar.selectbox(
            "Variance",
            [
                ("Auto detect", "auto"),
                ("Assume equal", True),
                ("Don't assume equal", False)
            ],
            format_func=lambda x: x[0]
        )
        
        if isinstance(self.equal_variance, str):
            self.equal_variance = self.equal_variance
        
        self.normality_check = st.sidebar.checkbox("Check normality", value=True)
        
        self.multiple_testing_correction = st.sidebar.selectbox(
            "Multiple Testing Correction",
            [
                ("Auto select", "auto"),
                ("FDR (Benjamini-Hochberg)", "fdr_bh"),
                ("FDR (Benjamini-Yekutieli)", "fdr_by"),
                ("Bonferroni", "bonferroni"),
                ("No correction", "none")
            ],
            format_func=lambda x: x[0]
        )[1]
        
        st.sidebar.subheader("🎯 Clustering")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            self.cluster_rows = st.checkbox("Cluster rows", value=False)
        with col2:
            self.cluster_cols = st.checkbox("Cluster columns", value=False)
        
        self.clustering_method = st.sidebar.selectbox(
            "Method",
            ["Hierarchical", "K-means"]
        ).lower()
        
        self.n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    
    def setup_visualization_widgets(self):
        """Setup visualization widgets"""
        st.sidebar.subheader("🎨 Visualization")
        
        self.palette_selector = st.sidebar.selectbox(
            "Color Palette",
            list(self.color_manager.palette_dict.keys()),
            index=6  # Coolwarm
        )
        
        self.center_colormap = st.sidebar.checkbox("Center at 0", value=False)
        
        self.custom_palette = st.sidebar.text_input(
            "Custom Palette (comma-separated HEX)",
            placeholder="#FF0000,#FFFF00,#00FF00"
        )
        
        st.sidebar.subheader("📊 Display")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            self.show_values = st.checkbox("Show values", value=True)
            self.x_axis_label = st.text_input("X-axis label", value="Samples")
        with col2:
            self.show_stat_annotations = st.checkbox("Statistical annotations", value=False)
            self.y_axis_label = st.text_input("Y-axis label", value="Genes")
        
        self.value_format = st.sidebar.selectbox(
            "Value format",
            ['.0f', '.1f', '.2f', '.3f', 'g', 'e'],
            index=1
        )
        
        self.text_color_mode = st.sidebar.selectbox(
            "Text color",
            [("Auto", "auto"), ("Always black", "black"), ("Always white", "white")],
            format_func=lambda x: x[0]
        )[1]
        
        self.colorbar_label = st.sidebar.text_input("Colorbar label", value="Expression")
    
    def parse_data_text(self, content, delimiter):
        """Parse data from text"""
        try:
            df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
            
            if len(df.columns) == 3:
                # Long format
                x_col, y_col, val_col = df.columns[:3]
                self.raw_data = df
                self.data_matrix = df.pivot(index=x_col, columns=y_col, values=val_col)
            elif len(df.columns) > 3:
                # Matrix format
                self.raw_data = df
                self.data_matrix = df.set_index(df.columns[0])
            else:
                st.error("Unsupported data format. Expected at least 3 columns.")
                return False
            
            # Save original data
            self.data_matrix_original = self.data_matrix.copy()
            
            st.success(f"✅ Data loaded! Shape: {self.data_matrix.shape}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error reading data: {e}")
            return False
    
    def detect_delimiter(self, content):
        """Detect delimiter"""
        lines = content.split('\n')[:10]
        delimiters = [',', ';', '\t', ' ']
        
        for delim in delimiters:
            if all(len(line.split(delim)) > 1 for line in lines if line.strip()):
                return delim
        
        return ','
    
    def load_test_data(self):
        """Load test data"""
        np.random.seed(42)
        
        # Generate realistic data
        genes = ['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D', 'Gene_E']
        samples = ['Control_1', 'Control_2', 'Treatment_1', 'Treatment_2', 'Treatment_3']
        
        data = []
        for i, gene in enumerate(genes):
            for j, sample in enumerate(samples):
                if 'Control' in sample:
                    value = np.random.normal(loc=10 + i*2, scale=1.5)
                else:
                    value = np.random.normal(loc=15 + i*3, scale=2.0)
                
                # Add some missing values
                if np.random.random() < 0.1:
                    value = np.nan
                
                data.append([gene, sample, round(value, 3)])
        
        df = pd.DataFrame(data, columns=['Gene', 'Sample', 'Expression'])
        
        # Parse as if from text
        csv_content = df.to_csv(index=False)
        self.parse_data_text(csv_content, ',')
        
        st.success("✅ Sample data loaded!")
    
    def perform_analysis(self):
        """Perform comprehensive analysis"""
        if self.data_matrix is None:
            st.error("❌ Please load data first")
            return
        
        with st.spinner("🔬 Performing analysis..."):
            # Clear warnings
            self.warning_system.warnings.clear()
            self.warning_system.recommendations.clear()
            
            # Step 1: Handle missing values
            raw_data = self.data_matrix_original.values
            
            # Check missing values
            missing_before = np.isnan(raw_data).sum()
            missing_percent = (missing_before / raw_data.size) * 100
            
            if missing_before > 0:
                st.info(f"Found missing values: {missing_before} ({missing_percent:.1f}%)")
            
            # Apply imputation
            imputed_data = self.missing_handler.handle(
                raw_data, 
                strategy=self.imputation_strategy
            )
            
            missing_after = np.isnan(imputed_data).sum()
            
            # Step 2: Normalization
            self.processed_data = self.normalization_engine.normalize(
                imputed_data,
                method=self.normalization_method
            )
            
            # Check normalization
            self.warning_system.check_normalization(imputed_data, self.normalization_method)
            
            # Save for display
            self.display_data = self.processed_data.copy()
            
            # Step 3: Descriptive statistics
            self._calculate_descriptive_stats()
            
            # Step 4: Statistical tests
            self._perform_statistical_tests()
            
            # Step 5: Clustering
            if self.cluster_rows or self.cluster_cols:
                self._perform_clustering()
            
            # Step 6: Check statistical power
            self._check_statistical_power()
            
            # Display warnings
            self.warning_system.display_warnings()
            
            st.session_state.analysis_done = True
            st.success("✅ Analysis completed!")
    
    def _calculate_descriptive_stats(self):
        """Calculate descriptive statistics"""
        data = self.processed_data
        
        stats_dict = {
            'Mean': np.nanmean(data),
            'Median': np.nanmedian(data),
            'Std': np.nanstd(data),
            'Min': np.nanmin(data),
            'Max': np.nanmax(data),
            '25% Quantile': np.nanpercentile(data, 25),
            '75% Quantile': np.nanpercentile(data, 75),
            'Skewness': stats.skew(data.flatten(), nan_policy='omit'),
            'Kurtosis': stats.kurtosis(data.flatten(), nan_policy='omit')
        }
        
        self.statistics_results['descriptive'] = stats_dict
        
        # Display in expander
        with st.expander("📊 Descriptive Statistics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                for key, value in list(stats_dict.items())[:5]:
                    st.metric(key, f"{value:.4f}")
            with col2:
                for key, value in list(stats_dict.items())[5:]:
                    st.metric(key, f"{value:.4f}")
    
    def _perform_statistical_tests(self):
        """Perform statistical tests"""
        # Auto-detect groups
        col_names = list(self.data_matrix.columns)
        groups = {}
        
        for name in col_names:
            name_lower = name.lower()
            if 'control' in name_lower or 'ctrl' in name_lower or 'normal' in name_lower:
                groups.setdefault('control', []).append(name)
            elif 'treat' in name_lower or 'exp' in name_lower or 'drug' in name_lower:
                groups.setdefault('treatment', []).append(name)
            elif 'group' in name_lower:
                import re
                numbers = re.findall(r'\d+', name)
                if numbers:
                    group_key = f'group_{numbers[0]}'
                    groups.setdefault(group_key, []).append(name)
        
        if len(groups) >= 2:
            # Take first two groups for comparison
            group_keys = list(groups.keys())[:2]
            group1_name, group2_name = group_keys
            
            # Get column indices
            group1_indices = [col_names.index(col) for col in groups[group1_name]]
            group2_indices = [col_names.index(col) for col in groups[group2_name]]
            
            # Determine if paired test
            paired = len(group1_indices) == len(group2_indices)
            
            # For each row perform test
            p_values = []
            test_stats = []
            test_names = []
            
            for i in range(self.processed_data.shape[0]):
                group1_data = self.processed_data[i, group1_indices]
                group2_data = self.processed_data[i, group2_indices]
                
                # Select test
                test_func, test_name, result = self.stat_test_selector.select_test(
                    group1_data, group2_data,
                    paired=paired,
                    equal_var=self.equal_variance,
                    normality_check=self.normality_check
                )
                
                if result:
                    stat, p_val = result
                    p_values.append(p_val)
                    test_stats.append(stat)
                    test_names.append(test_name)
                else:
                    p_values.append(np.nan)
                    test_stats.append(np.nan)
                    test_names.append('N/A')
            
            # Multiple testing correction
            pvals_clean = [p for p in p_values if not np.isnan(p)]
            if pvals_clean:
                pvals_corrected, rejected, method = self.stat_test_selector.apply_multiple_testing_correction(
                    pvals_clean,
                    method=self.multiple_testing_correction,
                    n_samples=self.processed_data.shape[0]
                )
                
                # Save results
                self.statistics_results['ttest'] = {
                    'p_values': p_values,
                    'p_values_corrected': pvals_corrected,
                    'test_stats': test_stats,
                    'test_names': test_names,
                    'significant': rejected,
                    'group1': group1_name,
                    'group2': group2_name,
                    'paired': paired,
                    'correction_method': method
                }
                
                # Display results
                with st.expander("🧪 Statistical Test Results", expanded=True):
                    st.write(f"**Groups:** {group1_name} vs {group2_name}")
                    st.write(f"**Test type:** {'Paired' if paired else 'Unpaired'}")
                    st.write(f"**Correction method:** {method}")
                    
                    sig_count = sum(rejected) if rejected is not None else 0
                    total = len(pvals_clean)
                    st.write(f"**Significant results (p<0.05):** {sig_count}/{total}")
                    
                    if sig_count > 0:
                        st.progress(sig_count / total)
            else:
                st.warning("Could not perform statistical tests")
        else:
            st.info("No groups detected for comparison")
    
    def _perform_clustering(self):
        """Perform clustering"""
        self.clustering_manager.cluster_data(
            self.processed_data,
            rows=self.cluster_rows,
            cols=self.cluster_cols,
            method=self.clustering_method,
            n_clusters=self.n_clusters
        )
        
        # Apply clustering to display data
        if self.cluster_rows and self.clustering_manager.row_order is not None:
            self.display_data, row_labels = self.clustering_manager.apply_clustering(
                self.display_data,
                labels=list(self.data_matrix.index),
                axis=0
            )
            self.data_matrix_display = self.data_matrix.iloc[self.clustering_manager.row_order]
        else:
            self.data_matrix_display = self.data_matrix.copy()
        
        if self.cluster_cols and self.clustering_manager.col_order is not None:
            self.display_data, col_labels = self.clustering_manager.apply_clustering(
                self.display_data,
                labels=list(self.data_matrix_display.columns),
                axis=1
            )
            self.data_matrix_display = self.data_matrix_display.iloc[:, self.clustering_manager.col_order]
        
        st.info("✅ Clustering applied")
    
    def _check_statistical_power(self):
        """Check statistical power"""
        n_samples = self.processed_data.shape[1]
        n_tests = self.processed_data.shape[0]
        
        if 'ttest' in self.statistics_results:
            n_tests = len([p for p in self.statistics_results['ttest']['p_values'] if not np.isnan(p)])
        
        self.warning_system.check_statistical_power(n_samples, n_tests, 'ttest')
    
    def create_visualization(self):
        """Create visualization"""
        if self.processed_data is None:
            st.error("❌ Please perform analysis first")
            return
        
        with st.spinner("🎨 Creating visualization..."):
            # Get data for display
            if hasattr(self, 'data_matrix_display'):
                data = self.display_data
                row_labels = list(self.data_matrix_display.index)
                col_labels = list(self.data_matrix_display.columns)
            else:
                data = self.processed_data
                row_labels = list(self.data_matrix.index)
                col_labels = list(self.data_matrix.columns)
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), 
                                   gridspec_kw={'width_ratios': [3, 1]})
            ax_main, ax_stats = axes
            
            # Get colormap
            cmap = self.color_manager.get_colormap(
                self.palette_selector,
                self.custom_palette
            )
            
            # Determine value range
            if self.center_colormap:
                vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
                vmin = -vmax
            else:
                vmin, vmax = np.nanmin(data), np.nanmax(data)
            
            # Check contrast
            contrast_problems = self.color_manager.check_color_contrast(
                cmap, data, self.center_colormap
            )
            if contrast_problems:
                st.warning("Possible text contrast issues")
            
            # Main heatmap
            im = ax_main.imshow(data, cmap=cmap, aspect='auto',
                              vmin=vmin, vmax=vmax)
            
            # Axis settings
            ax_main.set_xticks(np.arange(len(col_labels)))
            ax_main.set_yticks(np.arange(len(row_labels)))
            ax_main.set_xticklabels(col_labels, fontsize=10, rotation=45, ha='right')
            ax_main.set_yticklabels(row_labels, fontsize=10)
            ax_main.set_xlabel(self.x_axis_label, fontsize=12)
            ax_main.set_ylabel(self.y_axis_label, fontsize=12)
            
            # Add values
            if self.show_values:
                for i in range(len(row_labels)):
                    for j in range(len(col_labels)):
                        value = data[i, j]
                        
                        # Determine text to display
                        if self.show_stat_annotations and 'ttest' in self.statistics_results:
                            # Statistical annotations
                            p_vals = self.statistics_results['ttest']['p_values']
                            if i < len(p_vals) and not np.isnan(p_vals[i]):
                                p_val = p_vals[i]
                                if p_val < 0.001:
                                    text = '***'
                                elif p_val < 0.01:
                                    text = '**'
                                elif p_val < 0.05:
                                    text = '*'
                                else:
                                    text = format(value, self.value_format)
                            else:
                                text = format(value, self.value_format)
                        else:
                            # Regular values
                            text = format(value, self.value_format)
                        
                        # Determine text color
                        text_color = self.color_manager.get_text_color(
                            value, cmap, vmin, vmax, self.text_color_mode
                        )
                        
                        ax_main.text(j, i, text,
                                   ha='center', va='center',
                                   color=text_color,
                                   fontsize=9,
                                   fontweight='bold' if text in ['*', '**', '***'] else 'normal')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
            cbar.set_label(self.colorbar_label, fontsize=11)
            cbar.ax.tick_params(labelsize=9)
            
            # Title
            title_parts = []
            if self.normalization_method != 'none':
                title_parts.append(f"Norm: {self.normalization_method}")
            if self.imputation_strategy != 'none':
                title_parts.append(f"Impute: {self.imputation_strategy}")
            
            title = "Heatmap" + (" [" + ", ".join(title_parts) + "]" if title_parts else "")
            ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Statistics panel
            ax_stats.axis('off')
            stats_text = []
            
            if 'descriptive' in self.statistics_results:
                stats_text.append("📊 DESCRIPTIVE STATISTICS")
                stats = self.statistics_results['descriptive']
                for key in ['Mean', 'Median', 'Std', 'Min', 'Max']:
                    if key in stats:
                        stats_text.append(f"{key}: {stats[key]:.3f}")
            
            if 'ttest' in self.statistics_results:
                stats_text.append("\n🧪 STATISTICAL TESTS")
                ttest = self.statistics_results['ttest']
                stats_text.append(f"Groups: {ttest['group1']} vs {ttest['group2']}")
                stats_text.append(f"Type: {'Paired' if ttest['paired'] else 'Unpaired'}")
                
                if ttest['significant'] is not None:
                    sig_count = sum(ttest['significant'])
                    total = len([p for p in ttest['p_values'] if not np.isnan(p)])
                    stats_text.append(f"Significant: {sig_count}/{total}")
                
                if ttest['correction_method'] and ttest['correction_method'] != 'none':
                    stats_text.append(f"Correction: {ttest['correction_method']}")
            
            # Display statistics
            if stats_text:
                ax_stats.text(0.1, 0.95, '\n'.join(stats_text),
                            transform=ax_stats.transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Warnings
            if self.warning_system.warnings:
                warnings_text = ["⚠️ WARNINGS:"] + self.warning_system.warnings[:3]
                ax_stats.text(0.1, 0.4, '\n'.join(warnings_text),
                            transform=ax_stats.transAxes,
                            fontsize=8,
                            color='red',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
            
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
            
            # Export options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📥 Download Data (CSV)"):
                    self.download_data()
            
            with col2:
                if st.button("📊 Download Stats (CSV)"):
                    self.download_statistics()
            
            with col3:
                if st.button("🖼️ Download Plot (PNG)"):
                    self.download_plot(fig)
            
            with col4:
                if st.button("📝 Generate Report"):
                    self.generate_report()
            
            st.session_state.plot_created = True
    
    def reset_analysis(self):
        """Reset analysis"""
        self.processed_data = None
        self.display_data = None
        self.statistics_results = {}
        self.clustering_manager = ClusteringManager()
        
        st.session_state.data_loaded = False
        st.session_state.analysis_done = False
        st.session_state.plot_created = False
        
        st.rerun()
    
    def download_data(self):
        """Download processed data"""
        if self.processed_data is None:
            st.error("❌ No data to download")
            return
        
        df = pd.DataFrame(self.processed_data,
                         index=self.data_matrix.index,
                         columns=self.data_matrix.columns)
        
        csv = df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Click to download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def download_statistics(self):
        """Download statistics"""
        if not self.statistics_results:
            st.error("❌ No statistics to download")
            return
        
        # Create DataFrame with statistics
        stats_rows = []
        
        if 'descriptive' in self.statistics_results:
            for key, value in self.statistics_results['descriptive'].items():
                stats_rows.append(['Descriptive', key, value])
        
        if 'ttest' in self.statistics_results:
            ttest = self.statistics_results['ttest']
            for i, (p_val, p_corr) in enumerate(zip(ttest['p_values'], 
                                                   ttest.get('p_values_corrected', [None]*len(ttest['p_values'])))):
                if i < len(self.data_matrix.index):
                    row_name = self.data_matrix.index[i]
                else:
                    row_name = f'Row_{i+1}'
                
                stats_rows.append([
                    'T-test', row_name,
                    f"p={p_val:.4e}, p_corr={p_corr:.4e if p_corr else 'N/A'}, "
                    f"stat={ttest['test_stats'][i]:.4f}"
                ])
        
        df_stats = pd.DataFrame(stats_rows, columns=['Category', 'Item', 'Value'])
        csv = df_stats.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="statistics.csv">Click to download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def download_plot(self, fig):
        """Download plot as PNG"""
        from io import BytesIO
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="heatmap.png">Click to download PNG</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def generate_report(self):
        """Generate report"""
        if self.processed_data is None:
            st.error("❌ No analysis to report")
            return
        
        report = f"""
        ========================================
        DATA ANALYSIS REPORT
        ========================================
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. DATA
        ---------
        Size: {self.data_matrix.shape}
        Missing values: {np.isnan(self.data_matrix_original.values).sum()}
        Imputation strategy: {self.imputation_strategy}
        Normalization: {self.normalization_method}
        
        2. STATISTICS
        -------------
        """
        
        if 'descriptive' in self.statistics_results:
            report += "\nDescriptive statistics:\n"
            for key, value in self.statistics_results['descriptive'].items():
                report += f"  {key}: {value:.4f}\n"
        
        if 'ttest' in self.statistics_results:
            ttest = self.statistics_results['ttest']
            report += f"\nStatistical tests:\n"
            report += f"  Groups: {ttest['group1']} vs {ttest['group2']}\n"
            report += f"  Test type: {'Paired' if ttest['paired'] else 'Unpaired'}\n"
            if ttest['significant'] is not None:
                sig_count = sum(ttest['significant'])
                total = len([p for p in ttest['p_values'] if not np.isnan(p)])
                report += f"  Significant results: {sig_count}/{total}\n"
            if ttest['correction_method']:
                report += f"  Correction: {ttest['correction_method']}\n"
        
        report += "\n3. WARNINGS\n---------------\n"
        if self.warning_system.warnings:
            for warning in self.warning_system.warnings:
                report += f"  • {warning}\n"
        else:
            report += "  No warnings\n"
        
        report += "\n4. RECOMMENDATIONS\n--------------\n"
        if self.warning_system.recommendations:
            for rec in self.warning_system.recommendations:
                report += f"  • {rec}\n"
        else:
            report += "  No recommendations\n"
        
        # Display report
        st.text_area("📝 Analysis Report", report, height=300)
        
        # Download button
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="analysis_report.txt">Click to download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def run(self):
        """Run the application"""
        st.set_page_config(
            page_title="Scientific Heatmap Analyzer",
            page_icon="🔬",
            layout="wide"
        )
        
        # Header
        st.title("🔬 Scientific Heatmap Analyzer v2.0")
        st.markdown("""
        Statistical analysis and data visualization for scientific publications.
        
        **Main improvements in version 2.0:**
        1. 🛡️ Safe missing value handling (no fillna(0)!)
        2. 🧩 Modular architecture
        3. 🧪 Correct statistical tests with condition checking
        4. 🎯 Intelligent KMeans clustering
        5. ⚠️ Warning and recommendation system
        6. 🎨 Improved text color selection
        """)
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Main content area
        if st.session_state.data_loaded and hasattr(self, 'data_matrix_original') and self.data_matrix_original is not None:
            # Display data preview
            st.subheader("📋 Data Preview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data Matrix**")
                if self.data_matrix_original is not None:
                    st.dataframe(self.data_matrix_original.head(10), use_container_width=True)
                else:
                    st.info("No data loaded yet. Please load data using the sidebar.")
            
            with col2:
                if hasattr(self, 'processed_data'):
                    st.write("**Processed Data Matrix**")
                    display_df = pd.DataFrame(
                        self.processed_data,
                        index=self.data_matrix.index,
                        columns=self.data_matrix.columns
                    )
                    if hasattr(self, 'processed_data') and self.processed_data is not None:
                        display_df = pd.DataFrame(
                            self.processed_data,
                            index=self.data_matrix.index,
                            columns=self.data_matrix.columns
                        )
                        st.dataframe(display_df.head(10), use_container_width=True)
                    else:
                        st.info("Processed data will appear here after analysis.")
            
            # Display statistics if available
            if st.session_state.analysis_done:
                st.subheader("📊 Analysis Results")
                
                # Show data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Shape", f"{self.data_matrix.shape[0]} × {self.data_matrix.shape[1]}")
                with col2:
                    missing = np.isnan(self.data_matrix_original.values).sum()
                    st.metric("Missing Values", f"{missing}")
                with col3:
                    if 'ttest' in self.statistics_results and self.statistics_results['ttest']['significant'] is not None:
                        sig = sum(self.statistics_results['ttest']['significant'])
                        total = len([p for p in self.statistics_results['ttest']['p_values'] if not np.isnan(p)])
                        st.metric("Significant Tests", f"{sig}/{total}")

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    app = ScientificHeatmapApp()

    app.run()


