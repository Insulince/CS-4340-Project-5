import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public class Project5 {
    public static void wmain(String[] args) {
        double[][] XData = new double[][]{
                new double[]{LinearRegression.X_0, 1},
                new double[]{LinearRegression.X_0, 2},
                new double[]{LinearRegression.X_0, 3},
                new double[]{LinearRegression.X_0, 4},
                new double[]{LinearRegression.X_0, 5},
                new double[]{LinearRegression.X_0, 6},
                new double[]{LinearRegression.X_0, 7},
                new double[]{LinearRegression.X_0, 8},
        };

        SimpleMatrix XMatrix = new SimpleMatrix(XData);
        SimpleMatrix XTransposeMatrix = XMatrix.transpose();

        double[][] YData = new double[][]{
                new double[]{0},
                new double[]{1},
                new double[]{0},
                new double[]{1},
                new double[]{0},
                new double[]{1},
                new double[]{1},
                new double[]{1},
        };

        SimpleMatrix YMatrix = new SimpleMatrix(YData);

        SimpleMatrix XSwordMatrix = XTransposeMatrix.mult(XMatrix).invert().mult(XTransposeMatrix);

        System.out.println("XMatrix:\n" + XMatrix.toString());
        System.out.println("\nXTransposeMatrix:\n" + XTransposeMatrix.toString());
        System.out.println("\nYMatrix:\n" + YMatrix.toString());
        System.out.println("\nXSwordMatrix:\n" + XSwordMatrix.toString());
        System.out.println("\nXSword*y:\n" + XSwordMatrix.mult(YMatrix));
        System.out.println("\nXT*y:\n" + XTransposeMatrix.mult(YMatrix));
        System.out.println("\n(XT*X)^-1*XT*y:\n" + XSwordMatrix.mult(YMatrix));
        System.out.println("=========================");
        System.out.println(XTransposeMatrix.mult(XMatrix).mult(XSwordMatrix.mult(YMatrix)));
        System.out.println(XTransposeMatrix.mult(YMatrix));
        System.out.println("========================");
        System.out.println(XSwordMatrix.mult(YMatrix));

        SimpleMatrix finalSolution = XSwordMatrix.mult(YMatrix);
        double W_0 = finalSolution.get(0, 0);
        double W_1 = finalSolution.get(1, 0);
        System.out.println("W_0: " + W_0);
        System.out.println("W_1: " + W_1);

        System.out.println("y=" + W_1 + "*x+" + W_0);
    }

    private static final double RANDOM_DOUBLE_LOWER_BOUND = -2;
    private static final double RANDOM_DOUBLE_UPPER_BOUND = 10;

    static final int QUANTITY_DATA_SETS = 12;

    public static void main(final String[] args) throws Exception {
        final double[][] Xs = new double[QUANTITY_DATA_SETS][];
        for (int i = 0; i < Project5.QUANTITY_DATA_SETS; i++) {
            Xs[i] = new double[]{
                    LinearRegression.X_0,
                    Project5.randomDouble(Project5.RANDOM_DOUBLE_LOWER_BOUND, Project5.RANDOM_DOUBLE_UPPER_BOUND)
            };
        }

        final double[][] Ys = new double[QUANTITY_DATA_SETS][];
        for (int i = 0; i < Project5.QUANTITY_DATA_SETS; i++) {
            Ys[i] = new double[]{
                    randomSampleFunction(Xs[i][1])
            };
        }

        final LinearRegression linearRegression = new LinearRegression(Xs, Ys);
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
    private final double[][] Xs;
    private final double[][] Ys;
    private final double[] weights;
    private final int n;
    private final String regressionLine;
    private final String regularizedRegressionLine;

    LinearRegression(final double[][] Xs, final double[][] Ys) throws Exception {
        this.linearRegressionResult = new LinearRegressionResult();
        this.linearRegressionResult.setLambdas(LinearRegression.LAMBDAS);

        if (Xs.length == Ys.length) {
            this.Xs = Xs;
            this.linearRegressionResult.setXs(this.Xs);

            this.Ys = Ys;
            this.linearRegressionResult.setYs(this.Ys);

            this.weights = new double[Project5.QUANTITY_DATA_SETS];
            this.n = this.Xs.length;

            this.weights[0] = Math.random(); // Bias Weight
            this.weights[1] = Math.random(); // X_1 Weight

            this.regressionLine = this.calculateRegressionLine();
            this.linearRegressionResult.setRegressionLine(this.regressionLine);

            this.regularizedRegressionLine = this.calculateRegularizedRegression();
            this.linearRegressionResult.setRegularizedRegressionLine(this.regularizedRegressionLine);
        } else {
            throw new Exception("Number of Xs and number of Ys differ, cannot preform linear regression.");
        }
    }

    LinearRegressionResult getLinearRegressionResult() {
        return linearRegressionResult;
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "linearRegressionResult=" + this.linearRegressionResult +
                ", Xs=" + Arrays.toString(this.Xs) +
                ", Ys=" + Arrays.toString(this.Ys) +
                ", weights=" + Arrays.toString(this.weights) +
                ", n=" + this.n +
                ", regressionLine='" + this.regressionLine + '\'' +
                ", regularizedRegressionLine='" + this.regularizedRegressionLine + '\'' +
                '}';
    }

    private String calculateRegressionLine() {
        return "";
//        return this.calculate_X_sword().toString();
    }

    private String calculateRegularizedRegression() {
        System.out.println("Calculating regularized regression line...");

        String regularizedRegressionLine;

        double optimalLambda = this.calculateOptimalLambdaViaCrossValidation();
        this.linearRegressionResult.setFinalLambda(optimalLambda);

        final double final_E_in = this.calculate_E_aug(this.weights, this.Xs, this.Ys, optimalLambda);
        this.linearRegressionResult.setFinal_E_in(final_E_in);

        regularizedRegressionLine = "y=swag";
        System.out.println("Regularized regression line is: \"" + regularizedRegressionLine + "\"");
        return regularizedRegressionLine;
    }

    private double calculateOptimalLambdaViaCrossValidation() {
        System.out.println("--Calculate optimal lambda (min(for each lambda: E_cv(lambda))...");

        double smallest_E_cv = Double.MAX_VALUE;
        double optimalLambda = LinearRegression.LAMBDAS[0];

        for (int i = 0; i < LinearRegression.LAMBDAS.length; i++) {
            System.out.println("----Trying lambda \"" + LinearRegression.LAMBDAS[i] + "\"...");

            double current_E_cv = this.calculate_E_cv(this.weights, LinearRegression.LAMBDAS[i]);

            double[] temporaryResult_E_ins = this.linearRegressionResult.getE_ins();
            temporaryResult_E_ins[i] = this.calculate_E_in(this.weights, this.Xs, this.Ys);
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

    private double calculate_E_cv(final double[] weights, final double lambda) {
        System.out.println("------Calculating E_cv for lambda \"" + lambda + "\" ((sum(for each leaveOneOut set: leaveOneOut_E_aug(lambda))) / n)...");

        double E_cv;
        double[] allLeaveOneOut_E_aug = new double[this.n];

        for (int i = 0; i < this.n; i++) {
            final double[][] leaveOneOutXs = new double[this.n - 1][];
            final double[][] leaveOneOutYs = new double[this.n - 1][];

            int k = 0;
            for (int j = 0; j < this.n; j++) {
                if (i != j) {
                    leaveOneOutXs[k] = this.Xs[j];
                    leaveOneOutYs[k] = this.Ys[j];

                    k++;
                }
            }

            allLeaveOneOut_E_aug[i] = calculate_E_aug(weights, leaveOneOutXs, leaveOneOutYs, lambda);
        }

        Double sumLeaveOneOut_E_aug = 0.0;
        for (final Double leaveOneOut_E_aug : allLeaveOneOut_E_aug) {
            sumLeaveOneOut_E_aug += leaveOneOut_E_aug;
        }

        E_cv = sumLeaveOneOut_E_aug / this.n;
        System.out.println("------Calculated E_cv for lambda \"" + lambda + "\" to be \"" + E_cv + "\".");
        return E_cv;
    }

    private double calculate_E_aug(final double[] weights, final double[][] Xs, final double[][] Ys, final double lambda) {
        System.out.println("--------Calculating E_aug for lambda \"" + lambda + "\" (E_in + lambda * wTw)...");

        double E_aug;
        double wTw = 0.0;

        for (final double weight : weights) {
            wTw += Math.pow(weight, 2);
        }

        E_aug = this.calculate_E_in(weights, Xs, Ys) + lambda * wTw; // Ridge regression happens here.
        System.out.println("--------Calculated E_aug for lambda \"" + lambda + "\" to be \"" + E_aug + "\".");
        return E_aug;
    }

    private SimpleMatrix calculate_X_sword() {
        SimpleMatrix a = new SimpleMatrix(this.Xs);
        return a;
    }

    private double calculate_E_in(final double[] weights, final double[][] Xs, final double[][] Ys) {
        System.out.println("----------Calculating E_in ((sum((wTx - y)^2))/n)...");

        double E_in;
        final int n = Xs.length;

        double sumE_in = 0.0;

        for (int i = 0; i < n; i++) {
            final double X_i = Xs[i][1];
            final double Y_i = Ys[i][0];

            sumE_in += Math.pow((this.calculate_wTx(weights, X_i) - Y_i), 2);
        }

        E_in = sumE_in / n;
        System.out.println("----------Calculated E_in to be \"" + E_in + "\".");
        return E_in;
    }

    private double calculate_wTx(final double[] weights, final double X_1) {
        System.out.println("------------Calculating wTx (" + weights[0] + " * " + LinearRegression.X_0 + " + " + weights[1] + " * " + X_1 + ")...");

        double wTx = weights[0] * LinearRegression.X_0 + weights[1] * X_1;
        System.out.println("------------Calculated wTx to be \"" + wTx + "\".");
        return wTx;
    }
}

class LinearRegressionResult {
    private double[][] Xs;
    private double[][] Ys;
    private String regressionLine;
    private double[] lambdas;
    private double[] E_ins = new double[4];
    private double[] E_cvs = new double[4];
    private double finalLambda;
    private String regularizedRegressionLine;
    private double final_E_in;

    LinearRegressionResult() {
    }

    void setXs(final double[][] xs) {
        this.Xs = xs;
    }

    void setYs(final double[][] ys) {
        this.Ys = ys;
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
                "Xs=" + Arrays.toString(this.Xs) +
                ", Ys=" + Arrays.toString(this.Ys) +
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
        for (int i = 0; i < this.Xs.length; i++) {
            output.append("\n    • (").append(this.Xs[i][1]).append(", ").append(this.Ys[i][0]).append(")");
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
