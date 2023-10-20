#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import textwrap
from pathlib import Path

import libcst as cst
import libcst.matchers as m

description = """
Adds scope `if __name__ == "__main__":` to a python script immediately below the line
containing the comment `# INSERT MAIN`.

Used for converting .ipynb notebooks to main-guarded python scripts.
"""

INSERTION_POINT_TOKEN = "# MAIN"


class AddIfMainTransformer(cst.CSTTransformer):
    def __init__(self):
        self.split_stmt_idx = None
        self.statement_leadinglines_token_idx = None

    @staticmethod
    def make_ifmain_node(leading_lines, body_statements) -> cst.If:
        base_if_node = cst.parse_statement(
            textwrap.dedent(
                """if __name__ == "__main__": \n
                    pass"""
            )
        )
        base_if_node = base_if_node.with_changes(leading_lines=leading_lines)
        base_if_node = base_if_node.with_changes(
            body=cst.IndentedBlock(body=body_statements)
        )

        return base_if_node

    def visit_Module(self, node: cst.Module):
        comment_matcher = m.Comment(INSERTION_POINT_TOKEN)
        for i, child in enumerate(node.body):
            insertion_points = m.findall(child, comment_matcher)
            if len(insertion_points) > 0:
                matched_comment = insertion_points[0]
                self.split_stmt_idx = i
                self.statement_leadinglines_token_idx = [
                    e.comment for e in child.leading_lines
                ].index(matched_comment)
                # print(
                #     "Found at body node",
                #     i,
                #     "child leading line",
                #     self.statement_leadinglines_token_idx,
                # )
                break

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if self.split_stmt_idx is None or self.statement_leadinglines_token_idx is None:
            result_node = updated_node
        else:
            stmt_in_place = tuple(
                updated_node.body[i] for i in range(self.split_stmt_idx)
            )
            n_stmt_to_indent = len(updated_node.body) - self.split_stmt_idx
            stmt_to_indent = [
                updated_node.body[i]
                for i in range(self.split_stmt_idx + 1, len(updated_node.body))
            ]

            stmt_to_split = updated_node.body[self.split_stmt_idx]

            new_if_lls = stmt_to_split.leading_lines[
                : self.statement_leadinglines_token_idx + 1
            ]
            new_split_stmt_lls = stmt_to_split.leading_lines[
                self.statement_leadinglines_token_idx + 1 :
            ]
            stmt_to_split = stmt_to_split.with_changes(
                leading_lines=tuple(new_split_stmt_lls)
            )
            new_if_body_stmts = [stmt_to_split] + stmt_to_indent
            new_if_node = self.make_ifmain_node(new_if_lls, new_if_body_stmts)

            updated_node = updated_node.with_changes(
                body=tuple(stmt_in_place) + (new_if_node,)
            )
            result_node = updated_node

        return result_node


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent(description))
    parser.add_argument("-i", "--input", type=argparse.FileType("rt"), default="-")
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("wt+"), required=False, default="-"
    )
    args = parser.parse_args()

    in_stream = args.input

    in_script = in_stream.read()
    in_stream.close()
    cst_in = cst.parse_module(in_script)
    main_tf = AddIfMainTransformer()
    cst_out = cst_in.visit(main_tf)
    if not cst_in.deep_equals(cst_out):
        out_script = cst_out.code
    else:
        out_script = in_script

    out_stream = args.output
    out_stream.write(out_script)
    out_stream.close()
