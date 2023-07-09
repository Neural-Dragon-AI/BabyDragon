from babydragon.types import infer_embeddable_type
from typing import  List, Optional, Union
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.utils.main_logger import logger
from babydragon.utils.dataframes import extract_values_and_embeddings_pd, extract_values_and_embeddings_hf, extract_values_and_embeddings_polars, get_context_from_hf, get_context_from_pandas, get_context_from_polars
from babydragon.utils.pythonparser import extract_values_and_embeddings_python
import polars as pl
import os
import numpy as np
from pydantic import BaseModel
import libcst as cst

class DocstringCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.docstrings = []

    def visit_Module(self, node: cst.Module) -> bool:
        if node.body and isinstance(node.body[0].body, cst.SimpleStatementLine):
            for stmt in node.body[0].body.body:
                if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.SimpleString):
                    self.docstrings.append(stmt.value.value)
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        docstring = node.get_docstring()
        if docstring is not None:
            self.docstrings.append(docstring)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        docstring = node.get_docstring()
        if docstring is not None:
            self.docstrings.append(docstring)
        return True

    def collect(self):
        self.module.visit(self)
        return self.docstrings

class FunctionCallCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.function_calls = []

    def visit_Call(self, node: cst.Call) -> bool:
        if isinstance(node.func, cst.Name):
            # Add the function name to the list
            self.function_calls.append(node.func.value)
        return True

    def collect(self):
        self.module.visit(self)
        return self.function_calls

class ArgumentTypeCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.argument_types = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Collect argument types for functions
        arg_types = []
        for param in node.params.params:
            if isinstance(param.annotation, cst.Annotation):
                arg_types.append(param.annotation.annotation.value)
            else:
                arg_types.append(None)

        self.argument_types.append(arg_types)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Collect argument types for methods
        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                arg_types = []
                for param in stmt.params.params:
                    if isinstance(param.annotation, cst.Annotation):
                        arg_types.append(param.annotation.annotation.value)
                    else:
                        arg_types.append(None)

                self.argument_types.append(arg_types)

        return True

    def collect(self):
        self.module.visit(self)
        return self.argument_types

class ImportCollector(cst.CSTVisitor):
    def __init__(self, filename: str):
        with open(filename, "r") as file:
            self.module = cst.parse_module(file.read())
        self.imports = []

    def visit_Import(self, node: cst.Import) -> bool:
        for name in node.names:
            self.imports.append(cst.Module([node]))
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = node.module.value if node.module else ""
        for name in node.names:
            self.imports.append(cst.Module([node]))
        return True

    def collect(self):
        self.module.visit(self)
        import_code = [import_statement.code for import_statement in self.imports]
        return import_code

class IfStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.if_statements = []

    def visit_If(self, node: cst.If) -> bool:
        self.if_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.if_statements

class BaseCompoundStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.compound_statements = []

    def visit_BaseCompoundStatement(self, node: cst.BaseCompoundStatement) -> bool:
        self.compound_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.compound_statements
    
class ForLoopCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.for_loops = []

    def visit_For(self, node: cst.For) -> bool:
        self.for_loops.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.for_loops

class WhileLoopCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.while_loops = []

    def visit_While(self, node: cst.While) -> bool:
        self.while_loops.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.while_loops
    
class TryExceptCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.try_excepts = []

    def visit_Try(self, node: cst.Try) -> bool:
        self.try_excepts.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.try_excepts

class WithCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.withs = []

    def visit_With(self, node: cst.With) -> bool:
        self.withs.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.withs

class VariableDeclarationCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.variable_declarations = []

    def visit_Assign(self, node: cst.Assign) -> bool:
        self.variable_declarations.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.variable_declarations


class CodeFramePydantic(BaseModel):
    df_path: str
    context_columns: List
    embeddable_columns: List
    embedding_columns: List
    name: str
    save_path: Optional[str]
    save_dir: str
    text_embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder
    markdown: str

    class Config:
        arbitrary_types_allowed = True



class CodeFrame:
    def __init__(self, df: pl.DataFrame,
                context_columns: List = [],
                embeddable_columns: List = [],
                embedding_columns: List = [],
                name: str = "code_frame",
                save_path: Optional[str] = "/storage",
                text_embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder,
                markdown: str = "text/markdown",):
        self.df = df
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.embedding_columns = embedding_columns
        self.name = name
        self.save_path = save_path
        self.save_dir = f'{self.save_path}/{self.name}'
        self.text_embedder = text_embedder
        self.markdown = markdown
        self.frame_template = CodeFramePydantic(df_path=f'{self.save_dir}/{self.name}.parquet', context_columns=self.context_columns, embeddable_columns=self.embeddable_columns, embedding_columns=self.embedding_columns, name=self.name, save_path=self.save_path, save_dir=self.save_dir, load=True, text_embedder=self.text_embedder, markdown=self.markdown)


    def __getattr__(self, name: str):
        # delegate to the self.df object
        return getattr(self.df.lazy(), name)

    def get_overwritten_attr(self):
        df_methods = [method for method in dir(self.df) if callable(getattr(self.df, method))]
        memory_frame_methods = [method for method in dir(CodeFrame) if callable(getattr(CodeFrame, method))]
        common_methods = list(set(df_methods) & set(memory_frame_methods))
        return common_methods

    def embed_columns(self, embeddable_columns: List):
        for column_name in embeddable_columns:
            column = self.df[column_name]
            _, embedder = infer_embeddable_type(column)
            self._embed_column(column, embedder)

    def _embed_column(self, column, embedder):
        # Add the embeddings as a new column
        # Generate new values
        new_values = embedder.embed(self.df[column.name].to_list())
        # Add new column to DataFrame
        new_column_name = f'embedding|{column.name}'
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)
        self.embedding_columns.append(new_column_name)


    def search_column_with_sql_polar(self, sql_query, query, embeddable_column_name, top_k):
        df = self.df.filter(sql_query)
        embedding_column_name = 'embedding|' + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = df.with_columns(df[embedding_column_name].list.eval(pl.element().explode().dot(query_as_series),parallel=True).list.first().alias("dot_product"))
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result


    def search_column_polar(self, query, embeddable_column_name, top_k):
        embedding_column_name = 'embedding|' + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = self.df.with_columns(self.df[embedding_column_name].list.eval(pl.element().explode().dot(query_as_series),parallel=True).list.first().alias("dot_product"))
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result


    def save_parquet(self):
        #create dir in storage if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.full_save_path = f'{self.save_path}/{self.name}/{self.name}.parquet'
        self.df.write_parquet(self.full_save_path)
        frame_template_json = self.frame_template.json()
        with open(f'{self.save_dir}/{self.name}.json', 'w') as f:
            f.write(frame_template_json)

    @classmethod
    def load_parquet(cls, frame_path, name):
        df = pl.read_parquet(f'{frame_path}/{name}.parquet')
        with open(f'{frame_path}/{name}.json', 'r') as f:
            frame_template = CodeFramePydantic.parse_raw(f.read())
        return cls(df=df, context_columns=frame_template.context_columns, embeddable_columns=frame_template.embeddable_columns, embedding_columns=frame_template.embedding_columns, name=frame_template.name, save_path=frame_template.save_path, text_embedder=frame_template.text_embedder, markdown=frame_template.markdown)

    def generate_column(self, row_generator, new_column_name):
        # Generate new values
        new_values = row_generator.generate(self.df)
        # Add new column to DataFrame
        new_df = pl.DataFrame({ new_column_name: new_values })
        # Concatenate horizontally
        self.df = self.df.hstack([new_df])
    
    def apply_visitor_to_column(self, column_name: str, visitor_class: type):
        # Ensure the visitor_class is a subclass of PythonCodeVisitor
        if not issubclass(visitor_class, cst.CSTVisitor):
            raise TypeError('visitor_class must be a subclass of PythonCodeVisitor')

        # Iterate over the specified column
        new_values = []
        for code in self.df[column_name]:
            # Create a visitor and apply it to the code
            visitor = visitor_class(code)
            new_value = visitor.collect()
            new_values.append(new_value)
        # Generate new column
        new_column_name = f'{column_name}|{visitor_class.__name__}'
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)

        return self

    @classmethod
    def from_python(
        cls,
        directory_path: str,
        value_column: str,
        minify_code: bool = False,
        remove_docstrings: bool = False,
        resolution: str = "both",
        embeddings_column: List = [],
        embeddable_columns: List = [],
        context_columns: Optional[List[str]] = None,
        name: str = "code_frame",
        save_path: Optional[str] = "./storage",
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
        markdown: str = "text/markdown",
    ) -> "CodeFrame":
        values, context = extract_values_and_embeddings_python(directory_path, minify_code, remove_docstrings, resolution)
        logger.info(f"Found {len(values)} values in the directory {directory_path}")
        #convert retrieved data to polars dataframe
        df = pl.DataFrame({value_column: values})
        context_df = pl.DataFrame(context)
        #merge context columns with dataframe
        df = pl.concat([df, context_df], how='horizontal')
        if value_column not in embeddable_columns:
            embeddable_columns.append(value_column)
        return cls(df, context_columns, embeddable_columns, embeddings_column, name, save_path, embedder, markdown)