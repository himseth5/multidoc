from pydantic import BaseModel, ValidationError
from typing import List
from ast import literal_eval
from llama_index.core.output_parsers import ChainableOutputParser

class MetricResponse(BaseModel):
    """
    metric column is the name of the metric column
    columns is the list of columns that are used to form the metric column. it must include the actual name of table and not the alias.
    explanation is the string which explains the metric column, it must also include the calculation through which the metric column has been formed and by which table.
    """
    metric_column: str
    columns: List[str]
    explanation: str

class MetricList(BaseModel):
    """
    metrics is the list of MetricResponse Class
    """
    metrics: List[MetricResponse]

class CustomMetricListOutputParser(ChainableOutputParser):
    """Custom Metric List output parser.

    Assume first line is the name of the metric column.

    Assume second line as the list of columns used to create the above metric column.

    Assume third line is the explanation of the above metric column along with the calculation used.

    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse1(self, output: str) -> MetricList:
        """Parse output."""
        if self.verbose:
            print(f"> Raw output: {output}")
        lines = output.split("\n")
        metric_column = lines[0]
        columns = lines[1].split(",")
        explanation=lines[2]
        
        return MetricList(MetricResponse(metric_column=metric_column,columns=columns,explanation=explanation))
    
    def parse(self, output: str) -> MetricList:
        """Parse output."""
        if self.verbose:
            print(f"> Raw output: {output}")
            print(type(output))
        metrics = output.strip().split("\n")
        metric_list=[]
        print(metrics)
        for metric_string in metrics:
            metric_info=literal_eval(metric_string.strip(','))
            # metric_info = metric_info_tuple[0] if metric_info_tuple else []
            print("metric_info:", metric_info, type(metric_info))
            # lines = metric.strip(",").split()
            if isinstance(metric_info, tuple):
                metric_column = metric_info[0][0]
                print(metric_column)
                columns=metric_info[0][1]
                print(columns)
                explanation=metric_info[0][2]
                print(explanation)
            elif isinstance(metric_info, list):
                metric_column = metric_info[0]
                print(metric_column)
                columns=metric_info[1]
                print(columns)
                explanation=metric_info[2]
                print(explanation)
            metric_list.append(MetricResponse(metric_column=metric_column,columns=columns,explanation=explanation))

        
        return MetricList(metrics=metric_list)

    def parse2(self, output: str) -> MetricList:
        """Parse output."""
        if self.verbose:
            print(f"> Raw output: {output}")
            print(type(output))
        metrics = output.split("\n")
        metric_list=[]
        print(metrics)
        for metric_string in metrics:
            metric=literal_eval(metric_string)
            metric_column = metric[0]
            print(metric_column)
            columns=metric[1]
            print(columns)
            explanation=metric[2]
            print(explanation)
            metric_list.append(MetricResponse(metric_column=metric_column,columns=columns,explanation=explanation))

        
        return MetricList(metrics=metric_list)