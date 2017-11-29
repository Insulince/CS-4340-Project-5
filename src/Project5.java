import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public class Project5 {
    public static void smain(String[] args) {
        SimpleMatrix a = new SimpleMatrix(new double[][]{
                new double[]{1, 2, 3},
                new double[]{4, 5, 6},
                new double[]{7, 8, 9}
        });

        SimpleMatrix i = SimpleMatrix.identity(a.numRows());

        System.out.println(a);
        System.out.println(i);
        System.out.println(i.scale(5));
        System.out.println(a.plus(i.scale(5)));

//        double[][] XData = new double[][]{
//                new double[]{LinearRegression.X_0, 1},
//                new double[]{LinearRegression.X_0, 2},
//                new double[]{LinearRegression.X_0, 3},
//                new double[]{LinearRegression.X_0, 4},
//                new double[]{LinearRegression.X_0, 5},
//                new double[]{LinearRegression.X_0, 6},
//                new double[]{LinearRegression.X_0, 7},
//                new double[]{LinearRegression.X_0, 8},
//        };
//
//        SimpleMatrix X_matrix = new SimpleMatrix(XData);
//        SimpleMatrix XTransposeMatrix = X_matrix.transpose();
//
//        double[][] YData = new double[][]{
//                new double[]{0},
//                new double[]{1},
//                new double[]{0},
//                new double[]{1},
//                new double[]{0},
//                new double[]{1},
//                new double[]{1},
//                new double[]{1},
//        };
//
//        SimpleMatrix Y_matrix = new SimpleMatrix(YData);
//
//        SimpleMatrix XSwordMatrix = XTransposeMatrix.mult(X_matrix).invert().mult(XTransposeMatrix);
//
//        System.out.println("X_matrix:\n" + X_matrix.toString());
//        System.out.println("\nXTransposeMatrix:\n" + XTransposeMatrix.toString());
//        System.out.println("\nY_matrix:\n" + Y_matrix.toString());
//        System.out.println("\nXSwordMatrix:\n" + XSwordMatrix.toString());
//        System.out.println("\nXSword*y:\n" + XSwordMatrix.mult(Y_matrix));
//        System.out.println("\nXT*y:\n" + XTransposeMatrix.mult(Y_matrix));
//        System.out.println("\n(XT*X)^-1*XT*y:\n" + XSwordMatrix.mult(Y_matrix));
//        System.out.println("=========================");
//        System.out.println(XTransposeMatrix.mult(X_matrix).mult(XSwordMatrix.mult(Y_matrix)));
//        System.out.println(XTransposeMatrix.mult(Y_matrix));
//        System.out.println("========================");
//        System.out.println(XSwordMatrix.mult(Y_matrix));
//
//        SimpleMatrix finalSolution = XSwordMatrix.mult(Y_matrix);
//        double W_0 = finalSolution.get(0, 0);
//        double W_1 = finalSolution.get(1, 0);
//        System.out.println("W_0: " + W_0);
//        System.out.println("W_1: " + W_1);
//
//        System.out.println("y=" + W_1 + "*x+" + W_0);
    }

    private static final double RANDOM_DOUBLE_LOWER_BOUND = -2;
    private static final double RANDOM_DOUBLE_UPPER_BOUND = 10;

    static final int N = 12;
    static final int K = 3;

    public static void main(final String[] args) throws Exception {
        final double[][] XData = new double[N][];
        for (int i = 0; i < Project5.N; i++) {
            XData[i] = new double[]{
                    LinearRegression.X_0,
                    Project5.randomDouble(Project5.RANDOM_DOUBLE_LOWER_BOUND, Project5.RANDOM_DOUBLE_UPPER_BOUND)
            };
        }
        final SimpleMatrix X_matrix = new SimpleMatrix(XData);

        final double[][] YData = new double[N][];
        for (int i = 0; i < Project5.N; i++) {
            YData[i] = new double[]{
                    randomSampleFunction(XData[i][1])
            };
        }
        final SimpleMatrix Y_matrix = new SimpleMatrix(YData);

        final LinearRegression linearRegression = new LinearRegression(X_matrix, Y_matrix);
        System.out.println(linearRegression.getLinearRegressionResult().toResultString());
    }

    private static double randomDouble(final double lowerBoundInclusive, final double upperBoundInclusive) {
        return lowerBoundInclusive + (Math.random() * ((upperBoundInclusive - lowerBoundInclusive) + 1));
    }

    private static double randomSampleFunction(final double X) {
        return Math.pow(X, 2) + 10;
    }
}

class LinearRegression {
    static final double X_0 = 1.0;
    private static final double LAMBDA_1 = 0.1;
    private static final double LAMBDA_2 = 1.0;
    private static final double LAMBDA_3 = 10.0;
    private static final double LAMBDA_4 = 100.0;
    private static final double[] LAMBDAS = new double[]{LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4};

    private final LinearRegressionResult linearRegressionResult;
    private final SimpleMatrix X_matrix;
    private final SimpleMatrix Y_matrix;
    private final String regressionLine;
    private final String regularizedRegressionLine;

    LinearRegression(final SimpleMatrix X_matrix, final SimpleMatrix Y_matrix) throws Exception {
        this.linearRegressionResult = new LinearRegressionResult();
        this.linearRegressionResult.setLambdas(LinearRegression.LAMBDAS);

        if (X_matrix.numRows() == Y_matrix.numRows()) {
            this.X_matrix = X_matrix;
            this.linearRegressionResult.set__X_matrix(this.X_matrix);

            this.Y_matrix = Y_matrix;
            this.linearRegressionResult.set__Y_matrix(this.Y_matrix);

            this.regressionLine = this.calculateRegressionLine();
            this.linearRegressionResult.setRegressionLine(this.regressionLine);

            SimpleMatrix weightMatrix = new SimpleMatrix(new double[][]{
                    new double[]{
                            Math.random() // Bias Weight
                    },
                    new double[]{
                            Math.random() // X_1 Weight
                    }
            });
            this.regularizedRegressionLine = this.calculateRegularizedRegressionLine(weightMatrix);
            this.linearRegressionResult.setRegularizedRegressionLine(this.regularizedRegressionLine);
        } else {
            throw new Exception("Number of X_matrix and number of Y_matrix differ, cannot preform linear regression.");
        }
    }

    LinearRegressionResult getLinearRegressionResult() {
        return linearRegressionResult;
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "linearRegressionResult=" + this.linearRegressionResult +
                ", X_matrix=" + this.X_matrix +
                ", Y_matrix=" + this.Y_matrix +
                ", regressionLine='" + this.regressionLine + '\'' +
                ", regularizedRegressionLine='" + this.regularizedRegressionLine + '\'' +
                '}';
    }

    private String calculateRegressionLine() {
        SimpleMatrix XSwordMatrix = this.calculate_X_sword(this.X_matrix);
        SimpleMatrix W_lin = this.calculate_W_lin(XSwordMatrix, this.Y_matrix);

        return "y=" + W_lin.get(1, 0) + "*x+" + W_lin.get(0, 0);
    }

    private SimpleMatrix calculate_X_sword(final SimpleMatrix X_matrix) {
        return X_matrix.transpose().mult(X_matrix).invert().mult(X_matrix.transpose());
    }

    private SimpleMatrix calculate_W_lin(final SimpleMatrix XSwordMatrix, final SimpleMatrix Y_matrix) {
        return XSwordMatrix.mult(Y_matrix);
    }

    private String calculateRegularizedRegressionLine(final SimpleMatrix weightMatrix) {
        System.out.println("Calculating regularized regression line...");

        String regularizedRegressionLine;

        double optimalLambda = this.calculateOptimalLambdaViaCrossValidation(weightMatrix);
        this.linearRegressionResult.setFinalLambda(optimalLambda);

        final double final_E_in = this.calculate_E_aug(weightMatrix, this.X_matrix, this.Y_matrix, optimalLambda);
        this.linearRegressionResult.setFinal_E_in(final_E_in);

        SimpleMatrix regularized_X_sword_matrix = this.calculate_regularized_X_sword(this.X_matrix, optimalLambda);
        SimpleMatrix regularized_W_lin = this.calculate_W_lin(regularized_X_sword_matrix, this.Y_matrix);

        regularizedRegressionLine = "y=" + regularized_W_lin.get(1, 0) + "*x+" + regularized_W_lin.get(0, 0);

        System.out.println("Regularized regression line is: \"" + regularizedRegressionLine + "\"");
        return regularizedRegressionLine;
    }

    private SimpleMatrix calculate_regularized_X_sword(final SimpleMatrix X_matrix, final double lambda) {
        return X_matrix.transpose().mult(X_matrix).plus(SimpleMatrix.identity(X_matrix.numCols()).scale(lambda)).invert().mult(X_matrix.transpose());
    }

    private double calculateOptimalLambdaViaCrossValidation(final SimpleMatrix weightMatrix) {
        System.out.println("--Calculate optimal lambda (min(for each lambda: E_cv(lambda))...");

        double smallest_E_cv = Double.MAX_VALUE;
        double optimalLambda = LinearRegression.LAMBDAS[0];

        for (int i = 0; i < LinearRegression.LAMBDAS.length; i++) {
            System.out.println("----Trying lambda \"" + LinearRegression.LAMBDAS[i] + "\"...");

            double current_E_cv = this.calculate_E_cv(LinearRegression.LAMBDAS[i]);

            double[] temporaryResult_E_ins = this.linearRegressionResult.getE_ins();
            temporaryResult_E_ins[i] = this.calculate_E_aug(weightMatrix, this.X_matrix, this.Y_matrix, LinearRegression.LAMBDAS[i]);
            this.linearRegressionResult.setE_ins(temporaryResult_E_ins);

            double[] temporaryResult_E_Cvs = this.linearRegressionResult.getE_cvs();
            temporaryResult_E_Cvs[i] = current_E_cv;
            this.linearRegressionResult.setE_cvs(temporaryResult_E_Cvs);

            if (current_E_cv < smallest_E_cv) {
                System.out.println("----This lambda's E_cv (lambda: \"" + LinearRegression.LAMBDAS[i] + "\", E_cv: \"" + current_E_cv + "\") is better than the best lambda's E_cv so far (lambda: \"" + optimalLambda + "\", E_cv: \"" + smallest_E_cv + "\"), reassigning it.");

                smallest_E_cv = current_E_cv;
                optimalLambda = LinearRegression.LAMBDAS[i];
            } else {
                System.out.println("----This lambda's E_cv (lambda: \"" + LinearRegression.LAMBDAS[i] + "\", E_cv: \"" + current_E_cv + "\") is NOT better than the best lambda's E_cv so far (lambda: \"" + optimalLambda + "\", E_cv: \"" + smallest_E_cv + "\").");
            }
        }

        System.out.println("--Calculated optimal lambda to be \"" + optimalLambda + "\".");
        return optimalLambda;
    }

    private double calculate_E_cv(final double lambda) {
        System.out.println("------Calculating E_cv for lambda \"" + lambda + "\" ((sum(for each leaveNOut set: leaveNOut_E_aug(lambda))) / n)...");

        double E_cv;
        double[] all_E_val = new double[Project5.K];

        for (int i = 0; i < Project5.K; i++) {
            final double[][] leaveNInXData = new double[Project5.N - Project5.K][];
            final double[][] leaveNOutXData = new double[Project5.K][];
            final double[][] leaveNInYData = new double[Project5.N - Project5.K][];
            final double[][] leaveNOutYData = new double[Project5.K][];

            int k_in = 0;
            int k_out = 0;
            for (int j = 0; j < Project5.N; j++) {
                if (j < i * Project5.K || j >= (i + 1) * Project5.K) {
                    leaveNInXData[k_in] = new double[]{
                            this.X_matrix.get(j, 0),
                            this.X_matrix.get(j, 1)
                    };
                    leaveNInYData[k_in] = new double[]{
                            this.Y_matrix.get(j, 0)
                    };

                    k_in++;
                } else {
                    leaveNOutXData[k_out] = new double[]{
                            this.X_matrix.get(j, 0),
                            this.X_matrix.get(j, 1)
                    };
                    leaveNOutYData[k_out] = new double[]{
                            this.Y_matrix.get(j, 0)
                    };
                    k_out++;
                }
            }
            SimpleMatrix leaveNIn_X_matrix = new SimpleMatrix(leaveNInXData);
            SimpleMatrix leaveNIn_Y_matrix = new SimpleMatrix(leaveNInYData);
            SimpleMatrix leaveNOut_X_matrix = new SimpleMatrix(leaveNOutXData);
            SimpleMatrix leaveNOut_Y_matrix = new SimpleMatrix(leaveNOutYData);

            SimpleMatrix leaveNInXSwordMatrix = this.calculate_regularized_X_sword(leaveNIn_X_matrix, lambda);
            SimpleMatrix leaveNIn_W_lin = this.calculate_W_lin(leaveNInXSwordMatrix, leaveNIn_Y_matrix);

            all_E_val[i] = this.calculate_E_val(leaveNIn_W_lin, leaveNOut_X_matrix, leaveNOut_Y_matrix, lambda, Project5.K);
        }

        Double sum_E_val = 0.0;
        for (final double E_val : all_E_val) {
            sum_E_val += E_val;
        }

        E_cv = sum_E_val / Project5.K;
        System.out.println("------Calculated E_cv for lambda \"" + lambda + "\" to be \"" + E_cv + "\".");
        return E_cv;
    }

    private double calculate_E_val(final SimpleMatrix W_lin, final SimpleMatrix leaveNOut_X_matrix, final SimpleMatrix leaveNOut_Y_matrix, final double lambda, final int K) {
        return calculate_E_aug(W_lin, leaveNOut_X_matrix, leaveNOut_Y_matrix, lambda) / K;
    }

    private double calculate_E_aug(final SimpleMatrix weightMatrix, final SimpleMatrix X_matrix, final SimpleMatrix Y_matrix, final double lambda) {
        System.out.println("--------Calculating E_aug for lambda \"" + lambda + "\" (E_in + lambda * wTw)...");

        double E_aug;
        double wTw = 0.0;

        for (int i = 0; i < weightMatrix.numRows(); i++) {
            wTw += Math.pow(weightMatrix.get(i, 0), 2);
        }

        E_aug = this.calculate_E_in(weightMatrix, X_matrix, Y_matrix) + lambda * wTw; // Ridge regression happens here.
        System.out.println("--------Calculated E_aug for lambda \"" + lambda + "\" to be \"" + E_aug + "\".");
        return E_aug;
    }

    private double calculate_E_in(final SimpleMatrix weightMatrix, final SimpleMatrix X_matrix, final SimpleMatrix Y_matrix) {
        System.out.println("----------Calculating E_in ((sum((wTx - y)^2))/n)...");

        double E_in;

        double sumE_in = 0.0;

        for (int i = 0; i < X_matrix.numRows(); i++) {
            final double X_i = X_matrix.get(i, 1);
            final double Y_i = Y_matrix.get(i, 0);

            sumE_in += Math.pow((this.calculate_wTx(weightMatrix, X_i) - Y_i), 2);
        }

        E_in = sumE_in / Project5.N;
        System.out.println("----------Calculated E_in to be \"" + E_in + "\".");
        return E_in;
    }

    private double calculate_wTx(final SimpleMatrix weightMatrix, final double X_1) {
        System.out.println("------------Calculating wTx (" + weightMatrix.get(0, 0) + " * " + LinearRegression.X_0 + " + " + weightMatrix.get(1, 0) + " * " + X_1 + ")...");

        double wTx = weightMatrix.get(0, 0) * LinearRegression.X_0 + weightMatrix.get(1, 0) * X_1;
        System.out.println("------------Calculated wTx to be \"" + wTx + "\".");
        return wTx;
    }
}

class LinearRegressionResult {
    private SimpleMatrix X_matrix;
    private SimpleMatrix Y_matrix;
    private String regressionLine;
    private double[] lambdas;
    private double[] E_ins = new double[4];
    private double[] E_cvs = new double[4];
    private double finalLambda;
    private String regularizedRegressionLine;
    private double final_E_in;

    LinearRegressionResult() {
    }

    void set__X_matrix(final SimpleMatrix X_matrix) {
        this.X_matrix = X_matrix;
    }

    void set__Y_matrix(final SimpleMatrix Y_matrix) {
        this.Y_matrix = Y_matrix;
    }

    void setRegressionLine(final String regressionLine) {
        this.regressionLine = regressionLine;
    }

    void setLambdas(final double[] lambdas) {
        this.lambdas = lambdas;
    }

    double[] getE_ins() {
        return E_ins;
    }

    void setE_ins(final double[] E_ins) {
        this.E_ins = E_ins;
    }

    double[] getE_cvs() {
        return E_cvs;
    }

    void setE_cvs(final double[] E_cvs) {
        this.E_cvs = E_cvs;
    }

    void setFinalLambda(final double finalLambda) {
        this.finalLambda = finalLambda;
    }

    void setRegularizedRegressionLine(final String regularizedRegressionLine) {
        this.regularizedRegressionLine = regularizedRegressionLine;
    }

    void setFinal_E_in(final double final_E_in) {
        this.final_E_in = final_E_in;
    }

    @Override
    public String toString() {
        return "LinearRegressionResult{" +
                "_X_matrix=" + this.X_matrix +
                ", Y_matrix=" + this.Y_matrix +
                ", regressionLine='" + this.regressionLine + '\'' +
                ", lambdas=" + Arrays.toString(this.lambdas) +
                ", E_ins=" + Arrays.toString(this.E_ins) +
                ", E_cvs=" + Arrays.toString(this.E_cvs) +
                ", finalLambda=" + this.finalLambda +
                ", regularizedRegressionLine='" + this.regularizedRegressionLine + '\'' +
                ", final_E_in=" + this.final_E_in +
                '}';
    }

    String toResultString() {
        StringBuilder output = new StringBuilder("\n========================= RESULTS =========================");

        output.append("\n(a) Twelve (X, Y) coordinate pairs: ");
        for (int i = 0; i < Project5.N; i++) {
            output.append("\n    • (").append(this.X_matrix.get(i, 1)).append(", ").append(this.Y_matrix.get(i, 0)).append(")");
        }

        output.append("\n(b) Original Regression Line:");
        output.append("\n    • \"").append(this.regressionLine).append("\"");

        output.append("\n(c) Four (Lambda, E_in, E_cv) Triplets:");
        output.append("\n    • (").append(this.lambdas[0]).append(", ").append(this.E_ins[0]).append(", ").append(this.E_cvs[0]).append(")");
        output.append("\n    • (").append(this.lambdas[1]).append(", ").append(this.E_ins[1]).append(", ").append(this.E_cvs[1]).append(")");
        output.append("\n    • (").append(this.lambdas[2]).append(", ").append(this.E_ins[2]).append(", ").append(this.E_cvs[2]).append(")");
        output.append("\n    • (").append(this.lambdas[3]).append(", ").append(this.E_ins[3]).append(", ").append(this.E_cvs[3]).append(")");

        output.append("\n(d) Final Lambda:");
        output.append("\n    • ").append(this.finalLambda);

        output.append("\n(e) Regularized Regression Line:");
        output.append("\n    • \"").append(this.regularizedRegressionLine).append("\"");
        output.append("\n    Final E_in:");
        output.append("\n    • ").append(this.final_E_in);

        return output.toString();
    }
}
