import gradio as gr
import json

# Import the main analysis function
from prediction_pipeline import analyze_ticket, issue_type_model # Check if models loaded

def format_urgency_html(urgency_level: str) -> str:
    """Formats the urgency level with color using HTML."""
    color = "inherit" # Default text color
    icon = ""
    if urgency_level == "High":
        color = "#FF4136" # Red
        icon = "🔥 "
    elif urgency_level == "Medium":
        color = "#FF851B" # Orange
        icon = "⚠️ "
    elif urgency_level == "Low":
        color = "#2ECC40" # Green
        icon = "🟢 "
    
    if urgency_level == "N/A" or urgency_level == "Error":
         return f"<p style='font-size: 1.1em; font-weight: bold;'>{urgency_level}</p>"
         
    return f"<p style='color:{color}; font-size: 1.1em; font-weight: bold;'>{icon}{urgency_level}</p>"

def format_entities_html(entities_dict: dict) -> str:
    """Formats the entities dictionary into a more readable HTML string with pill-like styles."""
    if not entities_dict or not isinstance(entities_dict, dict):
        return "<div style='padding: 5px;'><p><em>No entities could be extracted or an error occurred.</em></p></div>"

    html_output = "<div style='padding: 5px;'>" 
    
    products = entities_dict.get("products", [])
    dates = entities_dict.get("dates", [])
    complaints = entities_dict.get("complaint_keywords", [])

    def create_pills(items_list, category_name):
        if not items_list:
            return f"<p style='margin-bottom: 8px;'><strong>{category_name}:</strong> <em>None identified</em></p>"
        
        pills_html = "".join([f"<span style='display: inline-block; background-color: #555; color: white; padding: 3px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em;'>{item}</span>" for item in items_list])
        return f"<p style='margin-bottom: 8px;'><strong>{category_name}:</strong><br>{pills_html}</p>"

    html_output += create_pills(products, "Products")
    html_output += create_pills(dates, "Dates")
    html_output += create_pills(complaints, "Complaint Keywords")
    html_output += "</div>"
    return html_output

def gradio_interface_for_blocks(ticket_text_input: str):
    """
    Wrapper function for Gradio interface.
    Takes raw text input and returns predictions and entities.
    Output order: issue_type (for gr.Label), urgency_html (for gr.HTML), entities_html (for gr.HTML)
    """
    if not isinstance(ticket_text_input, str) or not ticket_text_input.strip():
        return "N/A", format_urgency_html("N/A"), format_entities_html({})
        
    analysis_result = analyze_ticket(ticket_text_input)
    
    if "error" in analysis_result:
        error_message = analysis_result['error']
        return {"label": error_message, "__type__": "update"}, \
               format_urgency_html("Error"), \
               f"<div style='padding: 10px; color: red;'><strong>Error:</strong> {error_message}</div>"
        
    predicted_issue = analysis_result.get("predicted_issue_type", "Error")
    predicted_urgency = analysis_result.get("predicted_urgency_level", "Error")
    
    urgency_html = format_urgency_html(predicted_urgency)
    entities_html = format_entities_html(analysis_result.get("extracted_entities", {}))
    
    return predicted_issue, urgency_html, entities_html


# Check if models are loaded
if not issue_type_model:
    print("CRITICAL: Models were not loaded. The Gradio app might not function correctly.")
    print("Please ensure 'model_trainer.py' has been run successfully.")

# --- UI Definition with gr.Blocks ---
# Theme customization
# Using a Google Font and customizing hues for a professional look
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue, 
    secondary_hue=gr.themes.colors.sky, 
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="#f0f2f5", # Light grey background for the page
    block_background_fill="white", # White background for blocks/cards
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_hover="*neutral_300",
    input_background_fill="#ffffff",
)


with gr.Blocks(theme=theme, title="Customer Support Ticket Analyzer", css=".gradio-container { max-width: 1000px !important; margin: auto !important; }") as iface:
    # Section 1: Header (simulated with Markdown)
    gr.Markdown(
        """
        <div style="background-color: #2c3e50; color: white; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;">
            <h1 style="margin: 0; font-size: 1.5em;">🎫 Customer Support Ticket Analyzer</h1>
            <div style="font-size: 0.8em;">
                <!-- <a href="#" style="color: white; text-decoration: none; margin-left: 15px;">⚙️ Settings</a> -->
            </div>
        </div>
        """
    )
    gr.Markdown(
        """
        Enter a customer support ticket text below. The system will predict its issue type, urgency level, 
        and extract key entities. *(Ensure models are trained by running `model_trainer.py` first!)*
        """
    )

    # Section 2: Main Content Area
    with gr.Row(variant="panel", equal_height=False):
        # Left Panel: Ticket Input
        with gr.Column(scale=3):
            gr.Markdown("### Enter Customer Support Ticket")
            input_text = gr.Textbox(
                lines=12, 
                label=None, # Label provided by Markdown above
                placeholder="Paste your customer support ticket here...",
                elem_id="ticket_input_area" # For potential future CSS
            )
            gr.Markdown("<small>The system will predict its issue type and urgency level, and extract key entities like product names, dates, and complaint keywords.</small>")
            with gr.Row():
                clear_button = gr.Button("Clear Input", variant="secondary", size="sm")
                submit_button = gr.Button("Analyze Ticket", variant="primary", size="sm")
            
            with gr.Accordion("Example Tickets (Click to load)", open=False):
                gr.Examples(
                    examples=[
                        ["My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404 on 2023-03-15. This is a disaster."],
                        ["The AlphaWidget is not working as expected since 1st April 2024. It's showing a strange message and I am very unhappy."],
                        ["Everything is great with BetaService! Just a quick question about billing for May 2023."],
                        ["The new DeltaPlatform update from 2024-05-20 has made the system very slow and I am facing frequent crashes. This is unacceptable!"]
                    ],
                    inputs=input_text,
                    label=None, # Label provided by Accordion
                    elem_id="example_tickets_area"
                )
        
        # Right Panel: Analysis Results
        with gr.Column(scale=2):
            gr.Markdown("### Analysis Results")
            with gr.Group(): # Card 1: Predicted Issue Type
                gr.Markdown("🔧 **Predicted Issue Type**")
                output_issue_type = gr.Label(label=None, show_label=False) # Value will be prominent
            
            with gr.Group(): # Card 2: Predicted Urgency Level
                gr.Markdown("⚡ **Predicted Urgency Level**")
                output_urgency_level_html = gr.HTML() # For colored output
            
            with gr.Group(): # Card 3: Extracted Entities
                gr.Markdown("🏷️ **Extracted Entities**")
                output_entities_html = gr.HTML()

    # Footer
    gr.Markdown(
        """
        <hr style='margin-top: 30px; margin-bottom: 10px;'>
        <div style='text-align: center; font-size: 0.8em; color: #555;'>
            Built with <a href='https://www.gradio.app/' target='_blank' style='color: #2c3e50;'>Gradio</a> | 
            <a href='#' style='color: #2c3e50;'>Use via API (Placeholder)</a>
        </div>
        """
    )
    
    # Define button actions
    submit_button.click(
        fn=gradio_interface_for_blocks,
        inputs=input_text,
        outputs=[output_issue_type, output_urgency_level_html, output_entities_html]
    )
    
    def clear_all():
        # Clears input, issue_type label, urgency HTML, entities HTML
        return "", {"label": ""}, "", "" 

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, output_issue_type, output_urgency_level_html, output_entities_html]
    )

if __name__ == '__main__':
    print("Attempting to launch Gradio app with new professional UI...")
    if not issue_type_model:
         print("WARNING: Models not loaded. Gradio app predictions will likely fail or show 'N/A'.")
    iface.launch()
