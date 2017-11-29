import java.util.Arrays;

public class Project5 {
//    public static void smain(String[] args) {
//        double[][] a = new double[3][];
//
//        for (int i = 0; i < 3; i++) {
//            double[] a_i = new double[3];
//
//            for (int j = 0; j < 3; j++) {
//                a_i[j] = Math.floor(Math.random() * 10);
//            }
//
//            a[i] = a_i;
//        }
//
//        SimpleMatrix y = new SimpleMatrix(new double[][]{
//                new double[]{1.2, 21.2, 3.33, 4.23, 6.2289}
//        });
//
//        System.out.println((y.transpose().mult(y)).invert().mult(y.transpose()));
//    }

    private static final double RANDOM_DOUBLE_LOWER_BOUND = -2;
    private static final double RANDOM_DOUBLE_UPPER_BOUND = 10;

    static final int QUANTITY_DATA_SETS = 12;

    public static void main(final String[] args) throws Exception {
        final double[] Xs = new double[QUANTITY_DATA_SETS];
        for (int i = 0; i < Project5.QUANTITY_DATA_SETS; i++) {
            Xs[i] = Project5.randomDouble(Project5.RANDOM_DOUBLE_LOWER_BOUND, Project5.RANDOM_DOUBLE_UPPER_BOUND);
        }

        final double[] Ys = new double[QUANTITY_DATA_SETS];
        for (int i = 0; i < Project5.QUANTITY_DATA_SETS; i++) {
            Ys[i] = randomSampleFunction(Xs[i]);
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
    private static final double X_0 = 1.0;
    private static final int QUANTITY_LAMBDAS = 4;
    private static final double[] LAMBDAS = new double[LinearRegression.QUANTITY_LAMBDAS];
    private static final double LAMBDA_1 = 0.1;
    private static final double LAMBDA_2 = 1.0;
    private static final double LAMBDA_3 = 10.0;
    private static final double LAMBDA_4 = 100.0;

    static {
        LinearRegression.LAMBDAS[0] = LAMBDA_1;
        LinearRegression.LAMBDAS[1] = LAMBDA_2;
        LinearRegression.LAMBDAS[2] = LAMBDA_3;
        LinearRegression.LAMBDAS[3] = LAMBDA_4;
    }

    private final LinearRegressionResult linearRegressionResult;
    private final double[] Xs;
    private final double[] Ys;
    private final double[] weights;
    private final int n;
    private final int sumX;
    private final int sumY;
    private final int sumXY;
    private final int sumXSquared;
    private final double a;
    private final double b;
    private final String regressionLine;
    private final String regularizedRegressionLine;

    LinearRegression(final double[] Xs, final double[] Ys) throws Exception {
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

            this.sumX = this.calculateSumX();
            this.sumY = this.calculateSumY();
            this.sumXY = this.calculateSumXY();
            this.sumXSquared = this.calculateSumXSquared();

            this.a = this.calculateA();
            this.b = this.calculateB();

            this.regressionLine = "y=" + this.a + (this.b >= 0 ? "*x+" + this.b : "*x-" + -this.b);
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
                "linearRegressionResult=" + linearRegressionResult +
                ", Xs=" + Arrays.toString(this.Xs) +
                ", Ys=" + Arrays.toString(this.Ys) +
                ", weights=" + Arrays.toString(this.weights) +
                ", n=" + this.n +
                ", sumX=" + this.sumX +
                ", sumY=" + this.sumY +
                ", sumXY=" + this.sumXY +
                ", sumXSquared=" + this.sumXSquared +
                ", a=" + this.a +
                ", b=" + this.b +
                ", regressionLine='" + this.regressionLine + '\'' +
                ", regularizedRegressionLine='" + this.regularizedRegressionLine + '\'' +
                '}';
    }

    private int calculateSumX() {
        int sumX = 0;

        for (int i = 0; i < this.n; i++) {
            sumX += this.Xs[i];
        }

        return sumX;
    }

    private int calculateSumY() {
        int sumY = 0;

        for (int i = 0; i < this.n; i++) {
            sumY += this.Ys[i];
        }

        return sumY;
    }

    private int calculateSumXY() {
        int sumXY = 0;

        for (int i = 0; i < this.n; i++) {
            sumXY += this.Xs[i] * this.Ys[i];
        }

        return sumXY;
    }

    private int calculateSumXSquared() {
        int sumXSquared = 0;

        for (int i = 0; i < this.n; i++) {
            sumXSquared += Math.pow(this.Xs[i], 2);
        }

        return sumXSquared;
    }

    private double calculateA() {
        return (this.n * this.sumXY - this.sumX * this.sumY) / (this.n * this.sumXSquared - Math.pow(this.sumX, 2));
    }

    private double calculateB() {
        return (this.sumY - this.a * this.sumX) / this.n;
    }

    private String calculateRegularizedRegression() {
        System.out.println("Calculating regularized regression line...");

        String regularizedRegressionLine;

        double optimalLambda = this.calculateOptimalLambdaViaCrossValidation();
        this.linearRegressionResult.setFinalLambda(optimalLambda);

        final double final_E_in = this.calculate_E_aug(this.weights, this.Xs, this.Ys, optimalLambda);
        this.linearRegressionResult.setFinal_E_in(final_E_in);

        regularizedRegressionLine = "y=" + (this.a + optimalLambda) + (this.b >= 0 ? "*x+" + (this.b - optimalLambda) : "*x-" + -(this.b - optimalLambda));
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
            final double[] leaveOneOutXs = new double[this.n - 1];
            final double[] leaveOneOutYs = new double[this.n - 1];

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

    private double calculate_E_aug(final double[] weights, final double[] Xs, final double[] Ys, final double lambda) {
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

    private double calculate_E_in(final double[] weights, final double[] Xs, final double[] Ys) {
        System.out.println("----------Calculating E_in ((sum((wTx - y)^2))/n)...");

        double E_in;
        final int n = Xs.length;

        double sumE_in = 0.0;

        for (int i = 0; i < n; i++) {
            final double X_i = Xs[i];
            final double Y_i = Ys[i];

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
    private double[] Xs;
    private double[] Ys;
    private String regressionLine;
    private double[] lambdas;
    private double[] E_ins = new double[4];
    private double[] E_cvs = new double[4];
    private double finalLambda;
    private String regularizedRegressionLine;
    private double final_E_in;

    LinearRegressionResult() {
    }

    void setXs(final double[] xs) {
        Xs = xs;
    }

    void setYs(final double[] ys) {
        Ys = ys;
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
            output.append("\n    • (").append(this.Xs[i]).append(", ").append(this.Ys[i]).append(")");
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
