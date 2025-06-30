# Generated from waveforms/Waveform.g4 by ANTLR 4.11.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .WaveformParser import WaveformParser
else:
    from WaveformParser import WaveformParser

# This class defines a complete listener for a parse tree produced by WaveformParser.
class WaveformListener(ParseTreeListener):

    # Enter a parse tree produced by WaveformParser#expr.
    def enterExpr(self, ctx:WaveformParser.ExprContext):
        pass

    # Exit a parse tree produced by WaveformParser#expr.
    def exitExpr(self, ctx:WaveformParser.ExprContext):
        pass


    # Enter a parse tree produced by WaveformParser#assignment.
    def enterAssignment(self, ctx:WaveformParser.AssignmentContext):
        pass

    # Exit a parse tree produced by WaveformParser#assignment.
    def exitAssignment(self, ctx:WaveformParser.AssignmentContext):
        pass


    # Enter a parse tree produced by WaveformParser#PowerExpression.
    def enterPowerExpression(self, ctx:WaveformParser.PowerExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#PowerExpression.
    def exitPowerExpression(self, ctx:WaveformParser.PowerExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ConstantExpression.
    def enterConstantExpression(self, ctx:WaveformParser.ConstantExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ConstantExpression.
    def exitConstantExpression(self, ctx:WaveformParser.ConstantExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ListExpression.
    def enterListExpression(self, ctx:WaveformParser.ListExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ListExpression.
    def exitListExpression(self, ctx:WaveformParser.ListExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#TupleExpression.
    def enterTupleExpression(self, ctx:WaveformParser.TupleExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#TupleExpression.
    def exitTupleExpression(self, ctx:WaveformParser.TupleExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#NumberExpression.
    def enterNumberExpression(self, ctx:WaveformParser.NumberExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#NumberExpression.
    def exitNumberExpression(self, ctx:WaveformParser.NumberExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ShiftExpression.
    def enterShiftExpression(self, ctx:WaveformParser.ShiftExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ShiftExpression.
    def exitShiftExpression(self, ctx:WaveformParser.ShiftExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#FunctionCallExpression.
    def enterFunctionCallExpression(self, ctx:WaveformParser.FunctionCallExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#FunctionCallExpression.
    def exitFunctionCallExpression(self, ctx:WaveformParser.FunctionCallExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#IdentifierExpression.
    def enterIdentifierExpression(self, ctx:WaveformParser.IdentifierExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#IdentifierExpression.
    def exitIdentifierExpression(self, ctx:WaveformParser.IdentifierExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ParenthesesExpression.
    def enterParenthesesExpression(self, ctx:WaveformParser.ParenthesesExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ParenthesesExpression.
    def exitParenthesesExpression(self, ctx:WaveformParser.ParenthesesExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#UnaryMinusExpression.
    def enterUnaryMinusExpression(self, ctx:WaveformParser.UnaryMinusExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#UnaryMinusExpression.
    def exitUnaryMinusExpression(self, ctx:WaveformParser.UnaryMinusExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#MultiplyDivideExpression.
    def enterMultiplyDivideExpression(self, ctx:WaveformParser.MultiplyDivideExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#MultiplyDivideExpression.
    def exitMultiplyDivideExpression(self, ctx:WaveformParser.MultiplyDivideExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#AddSubtractExpression.
    def enterAddSubtractExpression(self, ctx:WaveformParser.AddSubtractExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#AddSubtractExpression.
    def exitAddSubtractExpression(self, ctx:WaveformParser.AddSubtractExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#StringExpression.
    def enterStringExpression(self, ctx:WaveformParser.StringExpressionContext):
        pass

    # Exit a parse tree produced by WaveformParser#StringExpression.
    def exitStringExpression(self, ctx:WaveformParser.StringExpressionContext):
        pass


    # Enter a parse tree produced by WaveformParser#NoArgFunction.
    def enterNoArgFunction(self, ctx:WaveformParser.NoArgFunctionContext):
        pass

    # Exit a parse tree produced by WaveformParser#NoArgFunction.
    def exitNoArgFunction(self, ctx:WaveformParser.NoArgFunctionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ArgsFunction.
    def enterArgsFunction(self, ctx:WaveformParser.ArgsFunctionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ArgsFunction.
    def exitArgsFunction(self, ctx:WaveformParser.ArgsFunctionContext):
        pass


    # Enter a parse tree produced by WaveformParser#KwargsFunction.
    def enterKwargsFunction(self, ctx:WaveformParser.KwargsFunctionContext):
        pass

    # Exit a parse tree produced by WaveformParser#KwargsFunction.
    def exitKwargsFunction(self, ctx:WaveformParser.KwargsFunctionContext):
        pass


    # Enter a parse tree produced by WaveformParser#ArgsKwargsFunction.
    def enterArgsKwargsFunction(self, ctx:WaveformParser.ArgsKwargsFunctionContext):
        pass

    # Exit a parse tree produced by WaveformParser#ArgsKwargsFunction.
    def exitArgsKwargsFunction(self, ctx:WaveformParser.ArgsKwargsFunctionContext):
        pass


    # Enter a parse tree produced by WaveformParser#args.
    def enterArgs(self, ctx:WaveformParser.ArgsContext):
        pass

    # Exit a parse tree produced by WaveformParser#args.
    def exitArgs(self, ctx:WaveformParser.ArgsContext):
        pass


    # Enter a parse tree produced by WaveformParser#kwargs.
    def enterKwargs(self, ctx:WaveformParser.KwargsContext):
        pass

    # Exit a parse tree produced by WaveformParser#kwargs.
    def exitKwargs(self, ctx:WaveformParser.KwargsContext):
        pass


    # Enter a parse tree produced by WaveformParser#kwarg.
    def enterKwarg(self, ctx:WaveformParser.KwargContext):
        pass

    # Exit a parse tree produced by WaveformParser#kwarg.
    def exitKwarg(self, ctx:WaveformParser.KwargContext):
        pass


    # Enter a parse tree produced by WaveformParser#list.
    def enterList(self, ctx:WaveformParser.ListContext):
        pass

    # Exit a parse tree produced by WaveformParser#list.
    def exitList(self, ctx:WaveformParser.ListContext):
        pass


    # Enter a parse tree produced by WaveformParser#tuple.
    def enterTuple(self, ctx:WaveformParser.TupleContext):
        pass

    # Exit a parse tree produced by WaveformParser#tuple.
    def exitTuple(self, ctx:WaveformParser.TupleContext):
        pass



del WaveformParser