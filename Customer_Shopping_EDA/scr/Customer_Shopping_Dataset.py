# Analysing customer shopping dataset by using the libraries include:
# Numpy, Pandas, Matplotlib and Seaborn.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerShoppingAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.prepare_data()

    def prepare_data(self):
        # Save CSV with error handling
        try:
            csv_path = r"C:\Users\Abhay\OneDrive\Documents\Customer_Shopping_Dataset_converted.csv"
            self.df.to_csv(csv_path, index=False)
            print(f"CSV file saved successfully at: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # Convert invoice_date to datetime
        self.df['invoice_date'] = pd.to_datetime(self.df['invoice_date'])

        # Create total purchase amount
        self.df['purchase_amount'] = self.df['quantity'] * self.df['price']

        # Create month column for trend analysis
        self.df['month'] = self.df['invoice_date'].dt.to_period('M')


    #  NUMPY ANALYSIS
    def numpy_analysis(self):
        purchase = self.df['purchase_amount'].values

        print("Mean Purchase:", np.mean(purchase))
        print("Median Purchase:", np.median(purchase))
        print("Standard Deviation:", np.std(purchase))

        print("Minimum Spend:", np.min(purchase))
        print("Maximum Spend:", np.max(purchase))

        print("25th Percentile:", np.percentile(purchase, 25))
        print("50th Percentile:", np.percentile(purchase, 50))
        print("75th Percentile:", np.percentile(purchase, 75))

        # Outlier thresholds (IQR method)
        q1 = np.percentile(purchase, 25)
        q3 = np.percentile(purchase, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        print("Outlier Lower Bound:", lower_bound)
        print("Outlier Upper Bound:", upper_bound)

    
    # PANDAS ANALYSIS
    def pandas_analysis(self):
        # Total revenue per category
        category_revenue = self.df.groupby('category')['purchase_amount'].sum()
        print("\nTotal Revenue per Category:\n", category_revenue)

        # Average spend per gender
        gender_avg = self.df.groupby('gender')['purchase_amount'].mean()
        print("\nAverage Spend per Gender:\n", gender_avg)

        # Purchase count per age group
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['Teen', 'Young Adult', 'Adult', 'Senior', 'Elder']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels)

        age_group_count = self.df.groupby('age_group', observed=True)['invoice_no'].count()
        print("\nPurchase Count per Age Group:\n", age_group_count)

        # Monthly revenue trend
        monthly_revenue = self.df.groupby('month')['purchase_amount'].sum()
        print("\nMonthly Revenue Trend:\n", monthly_revenue)

        # Shopping mall wise customer count
        mall_customers = self.df.groupby('shopping_mall')['customer_id'].nunique()
        print("\nCustomers per Shopping Mall:\n", mall_customers)
    

    # COMBINED VISUALS - ALL PLOTS IN ONE FIGURE
    def create_all_visualizations(self):
        # Create a figure with subplots (3 rows, 3 columns)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Customer Shopping Dataset - Complete Analysis', fontsize=20, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot 1: Monthly revenue trend
        monthly_revenue = self.df.groupby('month')['purchase_amount'].sum()
        monthly_revenue_df = monthly_revenue.reset_index()
        monthly_revenue_df['month'] = monthly_revenue_df['month'].astype(str)
        axes[0].plot(monthly_revenue_df['month'], monthly_revenue_df['purchase_amount'], marker='o', linewidth=2)
        axes[0].set_title("Monthly Revenue Trend", fontweight='bold')
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Revenue")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Revenue per category
        category_revenue = self.df.groupby('category')['purchase_amount'].sum().sort_values(ascending=False)
        axes[1].bar(range(len(category_revenue)), category_revenue.values, color='steelblue')
        axes[1].set_title("Revenue per Category", fontweight='bold')
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Revenue")
        axes[1].set_xticks(range(len(category_revenue)))
        axes[1].set_xticklabels(category_revenue.index, rotation=45, ha='right')
        
        # Plot 3: Purchase amount distribution (Histogram)
        axes[2].hist(self.df['purchase_amount'], bins=30, color='coral', edgecolor='black')
        axes[2].set_title("Purchase Amount Distribution", fontweight='bold')
        axes[2].set_xlabel("Purchase Amount")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: KDE plot
        sns.kdeplot(data=self.df['purchase_amount'], fill=True, ax=axes[3], color='purple')
        axes[3].set_title("KDE of Purchase Amount", fontweight='bold')
        axes[3].set_xlabel("Purchase Amount")
        axes[3].set_ylabel("Density")
        
        # Plot 5: Boxplot - purchase amount vs category
        sns.boxplot(x='category', y='purchase_amount', hue='category', data=self.df, ax=axes[4], palette='Set2', legend=False)
        axes[4].set_title("Outliers by Category", fontweight='bold')
        axes[4].set_xlabel("Category")
        axes[4].set_ylabel("Purchase Amount")
        axes[4].tick_params(axis='x', rotation=45)
        
        # Plot 6: Countplot - purchases by category
        sns.countplot(x='category', hue='category', data=self.df, ax=axes[5], palette='viridis', legend=False)
        axes[5].set_title("Purchase Count per Category", fontweight='bold')
        axes[5].set_xlabel("Category")
        axes[5].set_ylabel("Count")
        axes[5].tick_params(axis='x', rotation=45)
        
        # Plot 7: Correlation heatmap
        numeric_cols = ['age', 'quantity', 'price', 'purchase_amount']
        corr = self.df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[6], fmt='.2f', square=True)
        axes[6].set_title("Correlation Heatmap", fontweight='bold')
        
        # Plot 8: Gender-wise spending
        gender_revenue = self.df.groupby('gender')['purchase_amount'].sum()
        axes[7].pie(gender_revenue.values, labels=gender_revenue.index, autopct='%1.1f%%', 
                    startangle=90, colors=['skyblue', 'lightcoral'])
        axes[7].set_title("Revenue by Gender", fontweight='bold')
        
        # Plot 9: Age group distribution
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['Teen', 'Young Adult', 'Adult', 'Senior', 'Elder']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels)
        age_group_count = self.df.groupby('age_group', observed=True)['invoice_no'].count()
        axes[8].bar(age_group_count.index.astype(str), age_group_count.values, color='teal')
        axes[8].set_title("Purchases by Age Group", fontweight='bold')
        axes[8].set_xlabel("Age Group")
        axes[8].set_ylabel("Purchase Count")
        axes[8].tick_params(axis='x', rotation=45)
        
        # Apply tight_layout to prevent overlapping
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("Customer_Shopping_Visualization", )
        plt.show()


    # # INDIVIDUAL MATPLOTLIB VISUALS
    # def matplotlib_visuals(self):
    #     # Monthly revenue trend
    #     monthly_revenue = self.df.groupby('month')['purchase_amount'].sum()

    #     plt.figure()
    #     monthly_revenue.plot()
    #     plt.title("Monthly Revenue Trend")
    #     plt.xlabel("Month")
    #     plt.ylabel("Revenue")
    #     plt.tight_layout()
    #     plt.show()

    #     # Revenue per category
    #     category_revenue = self.df.groupby('category')['purchase_amount'].sum()

    #     plt.figure()
    #     category_revenue.plot(kind='bar')
    #     plt.title("Revenue per Category")
    #     plt.xlabel("Category")
    #     plt.ylabel("Revenue")
    #     plt.tight_layout()
    #     plt.show()

    #     # Purchase amount distribution
    #     plt.figure()
    #     plt.hist(self.df['purchase_amount'], bins=30)
    #     plt.title("Purchase Amount Distribution")
    #     plt.xlabel("Purchase Amount")
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.show()


    # # INDIVIDUAL SEABORN VISUALS
    # def seaborn_visuals(self):
    #     # KDE plot
    #     plt.figure()
    #     sns.kdeplot(self.df['purchase_amount'], fill=True)
    #     plt.title("KDE of Purchase Amount")
    #     plt.tight_layout()
    #     plt.show()

    #     # Boxplot: purchase amount vs category
    #     plt.figure()
    #     sns.boxplot(x='category', y='purchase_amount', data=self.df)
    #     plt.title("Outliers by Category")
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()

    #     # Countplot: purchases by category
    #     plt.figure()
    #     sns.countplot(x='category', data=self.df)
    #     plt.title("Purchase Count per Category")
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()

    #     # Correlation heatmap
    #     numeric_cols = ['age', 'quantity', 'price', 'purchase_amount']
    #     corr = self.df[numeric_cols].corr()

    #     plt.figure()
    #     sns.heatmap(corr, annot=True, cmap='coolwarm')
    #     plt.title("Correlation Heatmap")
    #     plt.tight_layout()
    #     plt.show()


file_path = r"C:\Users\Abhay\OneDrive\Documents\Customer_Shopping_Datasett.xlsx"
obj = CustomerShoppingAnalysis(file_path)
obj.prepare_data()
obj.numpy_analysis()
obj.pandas_analysis()
obj.create_all_visualizations()

# obj.matplotlib_visuals()
# obj.seaborn_visuals()
