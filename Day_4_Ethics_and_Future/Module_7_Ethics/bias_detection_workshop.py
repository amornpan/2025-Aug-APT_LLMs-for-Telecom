# Bias Detection and Ethical AI Assessment for Telecommunications
# Day 4 - Module 7 Practical Exercise

"""
This notebook provides tools for detecting and mitigating bias in LLM systems
specifically designed for telecommunications applications.
"""

# %% [markdown]
# ## 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 2. Sample Telecom Customer Service Dataset

# %%
# Create a sample dataset representing customer service interactions
def create_sample_dataset():
    """
    Create a sample dataset of customer service interactions
    with potential bias indicators
    """
    
    data = {
        'customer_id': range(1, 101),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], 100),
        'gender': np.random.choice(['Male', 'Female', 'Other'], 100, p=[0.45, 0.45, 0.1]),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100, p=[0.5, 0.3, 0.2]),
        'language': np.random.choice(['English', 'Spanish', 'Mandarin', 'Other'], 100, p=[0.6, 0.2, 0.1, 0.1]),
        'service_tier': np.random.choice(['Basic', 'Standard', 'Premium'], 100, p=[0.3, 0.5, 0.2]),
        'complaint_type': np.random.choice(['Billing', 'Technical', 'Service', 'Other'], 100),
        'resolution_time_hours': np.random.exponential(scale=5, size=100),
        'satisfaction_score': np.random.randint(1, 6, 100),
        'ai_response_quality': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some biases for demonstration
    # Premium customers get faster resolution
    df.loc[df['service_tier'] == 'Premium', 'resolution_time_hours'] *= 0.5
    
    # Urban customers get higher satisfaction
    df.loc[df['location'] == 'Urban', 'satisfaction_score'] += 1
    
    # Non-English speakers get lower AI response quality
    df.loc[df['language'] != 'English', 'ai_response_quality'] *= 0.8
    
    return df

df_customers = create_sample_dataset()
print("Sample Customer Service Dataset:")
print(df_customers.head())
print(f"\nDataset shape: {df_customers.shape}")

# %% [markdown]
# ## 3. Bias Detection Functions

# %%
class BiasDetector:
    """
    Class for detecting various types of bias in AI systems
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.bias_report = {}
    
    def demographic_parity(self, protected_attr: str, outcome: str, threshold: float = 0.1):
        """
        Check for demographic parity across protected attributes
        """
        groups = self.data.groupby(protected_attr)[outcome].mean()
        max_diff = groups.max() - groups.min()
        
        bias_detected = max_diff > threshold
        
        return {
            'attribute': protected_attr,
            'outcome': outcome,
            'group_means': groups.to_dict(),
            'max_difference': max_diff,
            'bias_detected': bias_detected,
            'threshold': threshold
        }
    
    def disparate_impact(self, protected_attr: str, outcome: str, threshold: float = 0.8):
        """
        Calculate disparate impact ratio
        """
        groups = self.data.groupby(protected_attr)[outcome].mean()
        
        if len(groups) == 2:
            ratio = min(groups) / max(groups)
            bias_detected = ratio < threshold
        else:
            ratio = None
            bias_detected = None
        
        return {
            'attribute': protected_attr,
            'outcome': outcome,
            'disparate_impact_ratio': ratio,
            'bias_detected': bias_detected,
            'threshold': threshold
        }
    
    def equal_opportunity(self, protected_attr: str, outcome: str, label: str):
        """
        Check for equal opportunity (equal true positive rates)
        """
        results = {}
        for group in self.data[protected_attr].unique():
            group_data = self.data[self.data[protected_attr] == group]
            positive_data = group_data[group_data[label] == 1]
            if len(positive_data) > 0:
                tpr = positive_data[outcome].mean()
                results[group] = tpr
        
        if results:
            max_diff = max(results.values()) - min(results.values())
            bias_detected = max_diff > 0.1
        else:
            max_diff = None
            bias_detected = None
        
        return {
            'attribute': protected_attr,
            'true_positive_rates': results,
            'max_difference': max_diff,
            'bias_detected': bias_detected
        }
    
    def statistical_parity_difference(self, protected_attr: str, outcome: str):
        """
        Calculate statistical parity difference
        """
        groups = self.data.groupby(protected_attr)[outcome].mean()
        overall_mean = self.data[outcome].mean()
        
        spd = {}
        for group, mean in groups.items():
            spd[group] = mean - overall_mean
        
        max_abs_diff = max(abs(v) for v in spd.values())
        bias_detected = max_abs_diff > 0.1
        
        return {
            'attribute': protected_attr,
            'outcome': outcome,
            'statistical_parity_diff': spd,
            'max_abs_difference': max_abs_diff,
            'bias_detected': bias_detected
        }
    
    def generate_bias_report(self, protected_attributes: List[str], outcomes: List[str]):
        """
        Generate comprehensive bias report
        """
        report = {
            'demographic_parity': [],
            'disparate_impact': [],
            'statistical_parity': []
        }
        
        for attr in protected_attributes:
            for outcome in outcomes:
                report['demographic_parity'].append(
                    self.demographic_parity(attr, outcome)
                )
                report['statistical_parity'].append(
                    self.statistical_parity_difference(attr, outcome)
                )
        
        self.bias_report = report
        return report

# %% [markdown]
# ## 4. Bias Detection Analysis

# %%
# Initialize bias detector
detector = BiasDetector(df_customers)

# Define protected attributes and outcomes
protected_attributes = ['gender', 'language', 'location', 'age_group']
outcomes = ['ai_response_quality', 'satisfaction_score']

# Detect bias for AI response quality across language groups
language_bias = detector.demographic_parity('language', 'ai_response_quality')
print("Language Bias in AI Response Quality:")
print(json.dumps(language_bias, indent=2))

# %%
# Visualize bias across different attributes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, attr in enumerate(protected_attributes):
    if idx < 4:
        group_means = df_customers.groupby(attr)['ai_response_quality'].mean()
        axes[idx].bar(group_means.index, group_means.values)
        axes[idx].set_title(f'AI Response Quality by {attr}')
        axes[idx].set_xlabel(attr)
        axes[idx].set_ylabel('Mean AI Response Quality')
        axes[idx].axhline(y=df_customers['ai_response_quality'].mean(), 
                         color='r', linestyle='--', label='Overall Mean')
        axes[idx].legend()
        axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Fairness Metrics Dashboard

# %%
def create_fairness_dashboard(data: pd.DataFrame, protected_attr: str, outcome: str):
    """
    Create a comprehensive fairness metrics dashboard
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fairness Dashboard: {outcome} across {protected_attr}', fontsize=16)
    
    # 1. Distribution plot
    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group][outcome]
        axes[0, 0].hist(group_data, alpha=0.5, label=group, bins=20)
    axes[0, 0].set_title('Distribution by Group')
    axes[0, 0].set_xlabel(outcome)
    axes[0, 0].legend()
    
    # 2. Box plot
    data.boxplot(column=outcome, by=protected_attr, ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot Comparison')
    
    # 3. Mean comparison
    means = data.groupby(protected_attr)[outcome].mean()
    axes[0, 2].bar(means.index, means.values)
    axes[0, 2].axhline(y=data[outcome].mean(), color='r', linestyle='--')
    axes[0, 2].set_title('Mean Comparison')
    axes[0, 2].set_xlabel(protected_attr)
    axes[0, 2].set_ylabel(f'Mean {outcome}')
    
    # 4. Cumulative distribution
    for group in data[protected_attr].unique():
        group_data = data[data[protected_attr] == group][outcome].sort_values()
        axes[1, 0].plot(group_data.values, 
                       np.arange(len(group_data))/len(group_data), 
                       label=group)
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel(outcome)
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].legend()
    
    # 5. Correlation matrix
    corr_data = pd.get_dummies(data[[protected_attr, outcome]])
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=axes[1, 1], cmap='coolwarm')
    axes[1, 1].set_title('Correlation Matrix')
    
    # 6. Fairness metrics summary
    detector_temp = BiasDetector(data)
    metrics = {
        'Demographic Parity': detector_temp.demographic_parity(protected_attr, outcome)['max_difference'],
        'Statistical Parity': detector_temp.statistical_parity_difference(protected_attr, outcome)['max_abs_difference']
    }
    
    axes[1, 2].bar(metrics.keys(), metrics.values())
    axes[1, 2].set_title('Fairness Metrics Summary')
    axes[1, 2].set_ylabel('Metric Value')
    axes[1, 2].axhline(y=0.1, color='r', linestyle='--', label='Threshold')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

# Create dashboard for language bias
create_fairness_dashboard(df_customers, 'language', 'ai_response_quality')

# %% [markdown]
# ## 6. Bias Mitigation Strategies

# %%
class BiasMitigation:
    """
    Implement various bias mitigation strategies
    """
    
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.mitigated_data = data.copy()
    
    def reweighting(self, protected_attr: str, outcome: str):
        """
        Apply reweighting to balance outcomes across groups
        """
        # Calculate weights to achieve demographic parity
        overall_mean = self.mitigated_data[outcome].mean()
        
        for group in self.mitigated_data[protected_attr].unique():
            group_mask = self.mitigated_data[protected_attr] == group
            group_mean = self.mitigated_data.loc[group_mask, outcome].mean()
            
            if group_mean != 0:
                weight = overall_mean / group_mean
                self.mitigated_data.loc[group_mask, 'weight'] = weight
            else:
                self.mitigated_data.loc[group_mask, 'weight'] = 1
        
        return self.mitigated_data
    
    def threshold_optimization(self, protected_attr: str, outcome: str, target: str):
        """
        Optimize decision thresholds for each group
        """
        thresholds = {}
        
        for group in self.mitigated_data[protected_attr].unique():
            group_data = self.mitigated_data[self.mitigated_data[protected_attr] == group]
            
            # Find optimal threshold for this group
            best_threshold = 0.5
            best_score = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                predictions = (group_data[outcome] > threshold).astype(int)
                accuracy = (predictions == group_data[target]).mean()
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        return thresholds
    
    def data_augmentation(self, protected_attr: str, minority_threshold: float = 0.2):
        """
        Augment data for underrepresented groups
        """
        value_counts = self.mitigated_data[protected_attr].value_counts(normalize=True)
        majority_size = len(self.mitigated_data) * value_counts.max()
        
        augmented_data = []
        
        for group, proportion in value_counts.items():
            if proportion < minority_threshold:
                group_data = self.mitigated_data[self.mitigated_data[protected_attr] == group]
                n_samples = int(majority_size - len(group_data))
                
                if n_samples > 0:
                    # Simple oversampling with noise
                    synthetic_samples = group_data.sample(n=n_samples, replace=True)
                    
                    # Add small noise to numerical columns
                    numerical_cols = synthetic_samples.select_dtypes(include=[np.number]).columns
                    for col in numerical_cols:
                        if col != 'customer_id':
                            noise = np.random.normal(0, 0.01, n_samples)
                            synthetic_samples[col] += noise
                    
                    augmented_data.append(synthetic_samples)
        
        if augmented_data:
            augmented_df = pd.concat([self.mitigated_data] + augmented_data, ignore_index=True)
            return augmented_df
        else:
            return self.mitigated_data
    
    def compare_before_after(self, protected_attr: str, outcome: str):
        """
        Compare bias metrics before and after mitigation
        """
        detector_before = BiasDetector(self.original_data)
        detector_after = BiasDetector(self.mitigated_data)
        
        before_metrics = detector_before.demographic_parity(protected_attr, outcome)
        after_metrics = detector_after.demographic_parity(protected_attr, outcome)
        
        comparison = {
            'attribute': protected_attr,
            'outcome': outcome,
            'before': {
                'max_difference': before_metrics['max_difference'],
                'bias_detected': before_metrics['bias_detected']
            },
            'after': {
                'max_difference': after_metrics['max_difference'],
                'bias_detected': after_metrics['bias_detected']
            },
            'improvement': before_metrics['max_difference'] - after_metrics['max_difference']
        }
        
        return comparison

# %% [markdown]
# ## 7. Apply Mitigation Strategies

# %%
# Apply mitigation
mitigator = BiasMitigation(df_customers)

# Apply reweighting
mitigated_data = mitigator.reweighting('language', 'ai_response_quality')

# Compare before and after
comparison = mitigator.compare_before_after('language', 'ai_response_quality')
print("Bias Mitigation Results:")
print(json.dumps(comparison, indent=2))

# Visualize improvement
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Before mitigation
groups_before = df_customers.groupby('language')['ai_response_quality'].mean()
ax1.bar(groups_before.index, groups_before.values)
ax1.axhline(y=df_customers['ai_response_quality'].mean(), color='r', linestyle='--')
ax1.set_title('Before Mitigation')
ax1.set_xlabel('Language')
ax1.set_ylabel('Mean AI Response Quality')

# After mitigation (simulated)
groups_after = mitigated_data.groupby('language')['ai_response_quality'].mean()
ax2.bar(groups_after.index, groups_after.values)
ax2.axhline(y=mitigated_data['ai_response_quality'].mean(), color='r', linestyle='--')
ax2.set_title('After Mitigation')
ax2.set_xlabel('Language')
ax2.set_ylabel('Mean AI Response Quality')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Compliance Checklist

# %%
class ComplianceChecker:
    """
    Check compliance with various regulations
    """
    
    def __init__(self):
        self.gdpr_checklist = {
            'data_minimization': False,
            'purpose_limitation': False,
            'consent_management': False,
            'right_to_explanation': False,
            'privacy_by_design': False,
            'data_portability': False,
            'right_to_erasure': False,
            'security_measures': False
        }
        
        self.ccpa_checklist = {
            'opt_out_mechanism': False,
            'non_discrimination': False,
            'disclosure_requirements': False,
            'deletion_rights': False,
            'data_sale_transparency': False
        }
        
        self.ai_ethics_checklist = {
            'bias_testing': False,
            'transparency': False,
            'human_oversight': False,
            'accountability': False,
            'fairness_metrics': False,
            'explainability': False
        }
    
    def check_gdpr_compliance(self, system_features: Dict[str, bool]):
        """
        Check GDPR compliance
        """
        for feature, implemented in system_features.items():
            if feature in self.gdpr_checklist:
                self.gdpr_checklist[feature] = implemented
        
        compliance_score = sum(self.gdpr_checklist.values()) / len(self.gdpr_checklist)
        return {
            'checklist': self.gdpr_checklist,
            'compliance_score': compliance_score,
            'compliant': compliance_score >= 0.8
        }
    
    def check_ai_ethics(self, system_features: Dict[str, bool]):
        """
        Check AI ethics compliance
        """
        for feature, implemented in system_features.items():
            if feature in self.ai_ethics_checklist:
                self.ai_ethics_checklist[feature] = implemented
        
        compliance_score = sum(self.ai_ethics_checklist.values()) / len(self.ai_ethics_checklist)
        return {
            'checklist': self.ai_ethics_checklist,
            'compliance_score': compliance_score,
            'ethical': compliance_score >= 0.8
        }
    
    def generate_compliance_report(self, system_features: Dict[str, bool]):
        """
        Generate comprehensive compliance report
        """
        gdpr_result = self.check_gdpr_compliance(system_features)
        ethics_result = self.check_ai_ethics(system_features)
        
        report = {
            'gdpr_compliance': gdpr_result,
            'ai_ethics_compliance': ethics_result,
            'overall_compliance': {
                'score': (gdpr_result['compliance_score'] + ethics_result['compliance_score']) / 2,
                'status': 'Compliant' if gdpr_result['compliant'] and ethics_result['ethical'] else 'Non-Compliant'
            }
        }
        
        return report

# %% [markdown]
# ## 9. Generate Compliance Report

# %%
# Example system features
system_features = {
    'data_minimization': True,
    'consent_management': True,
    'bias_testing': True,
    'transparency': True,
    'human_oversight': True,
    'fairness_metrics': True,
    'right_to_explanation': False,
    'privacy_by_design': True
}

# Check compliance
checker = ComplianceChecker()
compliance_report = checker.generate_compliance_report(system_features)

print("Compliance Report:")
print("="*50)
print(f"Overall Compliance Score: {compliance_report['overall_compliance']['score']:.2%}")
print(f"Status: {compliance_report['overall_compliance']['status']}")
print("\nGDPR Compliance:")
for item, status in compliance_report['gdpr_compliance']['checklist'].items():
    print(f"  {item}: {'✓' if status else '✗'}")
print(f"\nGDPR Score: {compliance_report['gdpr_compliance']['compliance_score']:.2%}")

print("\nAI Ethics Compliance:")
for item, status in compliance_report['ai_ethics_compliance']['checklist'].items():
    print(f"  {item}: {'✓' if status else '✗'}")
print(f"\nEthics Score: {compliance_report['ai_ethics_compliance']['compliance_score']:.2%}")

# %% [markdown]
# ## 10. Recommendations and Action Items

# %%
def generate_recommendations(bias_report: Dict, compliance_report: Dict):
    """
    Generate actionable recommendations based on bias and compliance analysis
    """
    recommendations = {
        'immediate_actions': [],
        'short_term': [],
        'long_term': []
    }
    
    # Check for bias issues
    if bias_report.get('bias_detected', False):
        recommendations['immediate_actions'].append(
            "Implement bias mitigation strategies for affected groups"
        )
        recommendations['short_term'].append(
            "Develop continuous bias monitoring system"
        )
    
    # Check compliance gaps
    gdpr_score = compliance_report['gdpr_compliance']['compliance_score']
    if gdpr_score < 0.8:
        recommendations['immediate_actions'].append(
            "Address GDPR compliance gaps immediately"
        )
    
    ethics_score = compliance_report['ai_ethics_compliance']['compliance_score']
    if ethics_score < 0.8:
        recommendations['short_term'].append(
            "Enhance AI ethics framework implementation"
        )
    
    # Long-term recommendations
    recommendations['long_term'].extend([
        "Establish AI governance committee",
        "Implement automated fairness testing pipeline",
        "Develop comprehensive AI ethics training program",
        "Create regular audit and review processes"
    ])
    
    return recommendations

# Generate recommendations
recommendations = generate_recommendations(
    language_bias,
    compliance_report
)

print("\n" + "="*50)
print("RECOMMENDATIONS")
print("="*50)

print("\n🚨 Immediate Actions:")
for action in recommendations['immediate_actions']:
    print(f"  • {action}")

print("\n📅 Short-term (1-3 months):")
for action in recommendations['short_term']:
    print(f"  • {action}")

print("\n🎯 Long-term (6-12 months):")
for action in recommendations['long_term']:
    print(f"  • {action}")

# %% [markdown]
# ## Summary
# 
# This notebook has demonstrated:
# 
# 1. **Bias Detection**: Multiple fairness metrics to identify bias in AI systems
# 2. **Visualization**: Comprehensive dashboards for understanding bias patterns
# 3. **Mitigation**: Practical strategies for reducing bias
# 4. **Compliance**: Checklists for GDPR and AI ethics compliance
# 5. **Recommendations**: Actionable steps for improvement
# 
# ### Key Takeaways:
# - Bias can occur across multiple dimensions (language, location, service tier)
# - Regular monitoring and testing is essential
# - Mitigation strategies must be tailored to specific use cases
# - Compliance requires systematic approach and documentation
# - Ethical AI is an ongoing process, not a one-time implementation
# 
# ### Next Steps:
# 1. Apply these techniques to your organization's LLM systems
# 2. Customize metrics for your specific use cases
# 3. Implement continuous monitoring
# 4. Establish governance frameworks
# 5. Train teams on ethical AI practices
