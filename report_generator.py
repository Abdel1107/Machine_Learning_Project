"""Kirollos Georgy - 40190279, Abdelaziz Mekkaoui - 40192247"""
"""Report Generator Module."""

from typing import Dict

from fpdf import FPDF


class ReportGenerator:
    """
    Class to generate a PDF report for the assignment.
    """

    def __init__(self, filename: str):
        """
        Initialize the PDF report with the given filename and write the start section.
        """
        self.__filename = filename
        self.__image_counter = 1
        self.__pdf = FPDF()
        self.__pdf.set_auto_page_break(auto=True, margin=15)
        self.__pdf.add_page()
        self.__pdf.set_font("Arial", style="B", size=16)
        self.__start_report()

    def write_section(
            self,
            metrics: Dict,
            confusion_matrix_file: str,
            roc_auc_file: str,
            section_heading: str,
            execution_time: float,
    ) -> None:
        """
        Write a section to the report with the graphs of that section.

        Args:
            metrics (Dict): Evaluation metrics of the model
            confusion_matrix_file (str): Path to the confusion matrix image
            roc_auc_file (str): Path to the ROC-AUC curve image
            section_heading (str): Heading of the section
            execution_time (float): Time taken to make predictions
        """
        self.__pdf.set_font("Arial", size=10, style="B")
        self.__pdf.cell(200, 10, txt=section_heading, ln=True, align="L")

        self.__pdf.set_font("Arial", size=10)
        self.__pdf.cell(
            200, 10, txt=f"Prediction Time: {execution_time} milliseconds", ln=True
        )
        self.__pdf.cell(200, 10, txt=f"Accuracy: {metrics['accuracy']}%", ln=True)
        self.__pdf.cell(200, 10, txt=f"Precision: {metrics['precision']}%", ln=True)
        self.__pdf.cell(200, 10, txt=f"Recall: {metrics['recall']}%", ln=True)
        self.__pdf.cell(200, 10, txt=f"F1 Score: {metrics['f1']}%", ln=True)
        self.__pdf.cell(200, 10, txt=f"ROC-AUC: {metrics['roc_auc']}%", ln=True)

        image_width = 110
        center_x = (self.__pdf.w - image_width) / 2
        self.__pdf.image(confusion_matrix_file, x=center_x, w=image_width)
        self.__pdf.cell(
            200,
            10,
            txt=f"Figure {self.__image_counter}: Confusion Matrix",
            ln=True,
            align="C",
        )
        self.__image_counter += 1
        center_x = (self.__pdf.w - image_width) / 2
        self.__pdf.image(roc_auc_file, x=center_x, w=image_width)
        self.__pdf.cell(
            200,
            10,
            txt=f"Figure {self.__image_counter}: ROC-AUC Curve",
            ln=True,
            align="C",
        )
        self.__image_counter += 1
        self.__pdf.add_page()

    def generate_file(self) -> None:
        """
        Write the report to the PDF file.
        """
        self.__document_end()
        self.__pdf.output(self.__filename)

    def __start_report(self) -> None:
        """
        Add the initial section to the report.
        """
        self.__pdf.set_font("Arial", style="B", size=16)
        self.__pdf.cell(200, 10, "COEN 432 Assignment 2 Report", ln=True, align="C")
        self.__pdf.ln(5)

        self.__pdf.set_font("Arial", size=10)
        self.__pdf.cell(200, 10, "Kirollos Georgy - 40190279", ln=True, align="C")
        self.__pdf.cell(200, 10, "Abdelaziz Mekkaoui - 40192247", ln=True, align="C")
        self.__pdf.ln(5)

        self.__pdf.set_font("Arial", size=12, style="B")
        self.__pdf.cell(200, 12, "Model and Data Handling", ln=True, align="L")

        self.__pdf.set_font("Arial", size=10)
        self.__pdf.multi_cell(
            0,
            10,
            (
                "In this project, we used the random forest classifier model for detecting the patients who have cancer. "
                "Random forest is basically a collection of decision trees, so it will be able to generalize as it picks the majority vote."
                "Since this is a complex problem with high-dimensional data, we chose this model because of its robustness and ability to handle non-linear data.\n"
                "To handle missing data, we used the KNN imputer. This imputer replaces missing values with the mean value of the k-nearest neighbors. "
                "This approach helps to maintain the integrity of the dataset and ensures that the model has complete data for training and testing.\n"
                "We also used a grid search to tune the hyperparameters of the random forest classifier to get the best possible parameters for the model. "
                "The grid search was performed using 10-fold cross-validation to evaluate the performance of the model on different subsets of the data. "
                "The evaluation metrics used in the grid search were accuracy, precision, recall, F1 score, and ROC AUC score. "
                "The best parameters found by the grid search were used to train the subsequent models with different training data sizes. \n"
                "Five different training set sizes were used to evaluate the performance of the model on different amounts of data. "
                "The model was trained on 40, 140, 240, 340, and 440 samples, and tested on 10, 35, 60, 85, and 110 respectively, keeping a ratio of 4 : 1 between "
                "the training and testing sets. This allowed us to assess how the model performs with varying amounts of training data."
            ),
        )
        self.__pdf.add_page()
        self.__pdf.set_font("Arial", size=12, style="B")
        self.__pdf.cell(200, 12, "Different Training Set Sizes", ln=True, align="L")

    def __document_end(self) -> None:
        """
        Add the final section to the report.
        """
        self.__pdf.set_font("Arial", size=12, style="B")
        self.__pdf.cell(200, 12, "Brief Analysis", ln=True, align="L")
        self.__pdf.set_font("Arial", size=10)
        self.__pdf.multi_cell(
            0,
            10,
            (
                "The model shows good performance with high accuracy, precision and ROC-AUC, indicating it is effective at identifying both positive and negative cases. "
                "Also, the recall and F1 score are also high (above 90%), suggesting that the model can detect most of the positive cases while maintaining a good balance between precision and recall. "
                "The confusion matrix and ROC-AUC curve provide visual representations of the model's performance, showing that it can distinguish between the two classes effectively "
                "with very few false positives and false negatives. "
                "Overall, the model is well-suited for cancer detection and can be used to help diagnose patients with high accuracy and reliability."
            ),
        )