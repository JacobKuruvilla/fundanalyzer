import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from text_analysis import TextAnalyzer
import io

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Funding Opportunities Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state if not exists
if 'text_analysis_ready' not in st.session_state:
    st.session_state.text_analysis_ready = False

# Initialize processors
@st.cache_resource
def get_processors():
    return DataProcessor(), TextAnalyzer()

data_processor, text_analyzer = get_processors()

# Main title
st.title("Funding Opportunities Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload Grants.gov Opportunity Search Excel file", type=['xlsx'])

if uploaded_file:
    try:
        # Load and process data
        data_processor.load_data(uploaded_file)

        # Top-level HHS division selector
        divisions = data_processor.get_hhs_divisions()
        selected_division = st.selectbox("Select HHS Division to Analyze", ["All Divisions"] + divisions)

        # Get opportunities for selected division
        opportunities = data_processor.get_division_opportunities(selected_division) if selected_division != "All Divisions" else data_processor.df

        # Initialize text analysis
        if not st.session_state.text_analysis_ready:
            text_analyzer.fit_transform_data(
                descriptions=opportunities['Description'].tolist(),
                eligibilities=opportunities['Eligibility'].tolist(),
                additional_eligibilities=opportunities['Additional Eligibility'].tolist(),
                opportunity_numbers=opportunities['Opportunity Number'].tolist(),
                assistance_listings=opportunities['Assistance Listings'].tolist()
            )
            st.session_state.text_analysis_ready = True

        # Add similarity analysis explanation
        st.header("Multi-Factor Similarity Analysis")
        with st.expander("ℹ️ Understanding the Similarity Analysis"):
            st.markdown("""
            This analysis uses an advanced approach combining vector-based and semantic matching:

            1. **Vector-Based Analysis** (70% total):
               - Description Similarity (35%)
               - Eligibility Matching (25%)
               - Additional Requirements (10%)

            2. **Semantic Analysis** (30%):
               - Deep semantic understanding using Claude AI
               - Concept matching and thematic analysis
               - Context-aware similarity scoring
            """)

        # Process opportunities
        for _, opportunity in opportunities.iterrows():
            opp_details = data_processor.get_opportunity_details(opportunity['Opportunity Number'])

            # Get similar opportunities with combined scoring
            similar_opps = text_analyzer.find_similar_opportunities(
                query_desc=opp_details['description'],
                query_elig=opp_details.get('eligibility', ''),
                query_add_elig=opp_details.get('additional_eligibility', ''),
                current_opp_number=opp_details['number']
            )

            if similar_opps:
                with st.expander(f"### {opportunity['Opportunity Number']} - {opportunity['Opportunity Title']}"):
                    # Show source opportunity details
                    st.markdown("#### Source Opportunity Details")
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Description**")
                        st.markdown(opp_details['description'][:300] + "..." if len(opp_details['description']) > 300 else opp_details['description'])

                    with cols[1]:
                        st.markdown("**Eligibility Information**")
                        st.markdown(f"*Basic Eligibility:* {opp_details.get('eligibility', 'Not specified')}")
                        st.markdown(f"*Additional Requirements:* {opp_details.get('additional_eligibility', 'Not specified')}")
                        st.markdown(f"*Program Number:* {opp_details.get('assistance_listings', 'Not specified')}")

                    # Create similarity matches table
                    st.markdown("#### Similar Opportunities")
                    similar_data = []
                    for idx, similarity_scores in similar_opps[:10]:  # Show top 10 matches
                        similar_opp = data_processor.df.iloc[idx]
                        similar_opp_details = data_processor.get_opportunity_details(similar_opp['Opportunity Number'])

                        similar_data.append({
                            'Opportunity Number': similarity_scores['opportunity_number'],
                            'Title': similar_opp['Opportunity Title'],
                            'HHS Division': similar_opp_details['hhs_division'],
                            'Description Match': f"{similarity_scores['description_similarity']*100:.1f}%",
                            'Eligibility Match': f"{similarity_scores['eligibility_similarity']*100:.1f}%",
                            'Additional Eligibility Match': f"{similarity_scores['additional_eligibility_similarity']*100:.1f}%",
                            'Semantic Match': f"{similarity_scores['semantic_similarity']*100:.1f}%",
                            'Overall Match': f"{similarity_scores['combined_score']*100:.1f}%",
                            'Program Number': similarity_scores['assistance_listing'],
                            'Close Date': similar_opp_details['close_date']
                        })

                        # Display semantic explanation
                        st.markdown(f"**Semantic Analysis: {similar_opp['Opportunity Number']}**")
                        st.markdown(similarity_scores['semantic_explanation'])

                    if similar_data:
                        st.dataframe(
                            pd.DataFrame(similar_data),
                            column_config={
                                'Opportunity Number': st.column_config.TextColumn('Number'),
                                'Title': st.column_config.TextColumn('Title', width='large'),
                                'HHS Division': st.column_config.TextColumn('Division'),
                                'Description Match': st.column_config.TextColumn('Description'),
                                'Eligibility Match': st.column_config.TextColumn('Eligibility'),
                                'Additional Eligibility Match': st.column_config.TextColumn('Add. Eligibility'),
                                'Semantic Match': st.column_config.TextColumn('Semantic'),
                                'Overall Match': st.column_config.TextColumn('Overall Score'),
                                'Program Number': st.column_config.TextColumn('Program Number'),
                                'Close Date': st.column_config.TextColumn('Closes')
                            }
                        )

        # Export functionality
        st.markdown("---")
        st.subheader("Export Analysis")

        # Create sheets for Excel export
        excel_data = {
            'Selected Division Opportunities': opportunities,
            'Similarity Analysis': []
        }

        # Collect data for similarity analysis
        similarity_rows = []

        for _, opportunity in opportunities.iterrows():
            opp_details = data_processor.get_opportunity_details(opportunity['Opportunity Number'])
            similar_opps = text_analyzer.find_similar_opportunities(
                query_desc=opp_details['description'],
                query_elig=opp_details.get('eligibility', ''),
                query_add_elig=opp_details.get('additional_eligibility', ''),
                current_opp_number=opp_details['number']
            )

            for idx, similarity_scores in similar_opps:
                similar_opp = data_processor.df.iloc[idx]
                similar_opp_details = data_processor.get_opportunity_details(similar_opp['Opportunity Number'])

                # Add similarity analysis row
                similarity_rows.append({
                    'Source Opportunity': opportunity['Opportunity Number'],
                    'Source Title': opportunity['Opportunity Title'],
                    'Source HHS Division': opp_details['hhs_division'],
                    'Source Program Number': opp_details['assistance_listings'],
                    'Similar Opportunity': similarity_scores['opportunity_number'],
                    'Similar Title': similar_opp['Opportunity Title'],
                    'Similar HHS Division': similar_opp_details['hhs_division'],
                    'Similar Program Number': similarity_scores['assistance_listing'],
                    'Description Similarity': f"{similarity_scores['description_similarity']*100:.1f}%",
                    'Eligibility Similarity': f"{similarity_scores['eligibility_similarity']*100:.1f}%",
                    'Additional Eligibility Similarity': f"{similarity_scores['additional_eligibility_similarity']*100:.1f}%",
                    'Semantic Similarity': f"{similarity_scores['semantic_similarity']*100:.1f}%",
                    'Combined Score': f"{similarity_scores['combined_score']*100:.1f}%",
                    'Semantic Analysis': similarity_scores['semantic_explanation']
                })

        # Update excel_data with collected rows
        excel_data['Similarity Analysis'] = pd.DataFrame(similarity_rows)

        if st.button("📥 Export to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            output.seek(0)
            st.download_button(
                label="💾 Download Excel file",
                data=output,
                file_name=f"funding_analysis_{selected_division.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload an Excel file to begin analysis.")
