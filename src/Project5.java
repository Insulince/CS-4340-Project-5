import java.util.ArrayList;

public class Project5 {
    private static final int QUANTITY_RANDOM_DOUBLES = 12;
    private static final double RANDOM_DOUBLE_LOWER_BOUND = -2;
    private static final double RANDOM_DOUBLE_UPPER_BOUND = 10;

    public static void main(final String[] args) throws Exception {
        final ArrayList<Double> Xs = new ArrayList<>();
        for (int i = 0; i < Project5.QUANTITY_RANDOM_DOUBLES; i++) {
            Xs.add(Project5.randomDouble(Project5.RANDOM_DOUBLE_LOWER_BOUND, Project5.RANDOM_DOUBLE_UPPER_BOUND));
        }

        final ArrayList<Double> Ys = new ArrayList<>();
        for (final double X : Xs) {
            Ys.add(randomSampleFunction(X));
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
    private static final ArrayList<Double> LAMBDAS = new ArrayList<>();
    private static final double LAMBDA_1 = 0.1;
    private static final double LAMBDA_2 = 1.0;
    private static final double LAMBDA_3 = 10.0;
    private static final double LAMBDA_4 = 100.0;

    static {
        LinearRegression.LAMBDAS.add(LAMBDA_1);
        LinearRegression.LAMBDAS.add(LAMBDA_2);
        LinearRegression.LAMBDAS.add(LAMBDA_3);
        LinearRegression.LAMBDAS.add(LAMBDA_4);
    }

    private final LinearRegressionResult linearRegressionResult;
    private final ArrayList<Double> Xs;
    private final ArrayList<Double> Ys;
    private final ArrayList<Double> weights;
    private final int n;
    private final int sumX;
    private final int sumY;
    private final int sumXY;
    private final int sumXSquared;
    private final double a;
    private final double b;
    private final String regressionLine;
    private final String regularizedRegressionLine;

    LinearRegression(final ArrayList<Double> Xs, final ArrayList<Double> Ys) throws Exception {
        this.linearRegressionResult = new LinearRegressionResult();
        this.linearRegressionResult.setLambdas(LinearRegression.LAMBDAS);

        if (Xs.size() == Ys.size()) {
            this.Xs = Xs;
            this.linearRegressionResult.setXs(this.Xs);

            this.Ys = Ys;
            this.linearRegressionResult.setYs(this.Ys);

            this.weights = new ArrayList<>();
            this.n = this.Xs.size();

            this.weights.add(Math.random()); // Bias Weight
            this.weights.add(Math.random()); // X 1 Weight

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
                ", Xs=" + this.Xs +
                ", Ys=" + this.Ys +
                ", weights=" + this.weights +
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
            sumX += this.Xs.get(i);
        }

        return sumX;
    }

    private int calculateSumY() {
        int sumY = 0;

        for (int i = 0; i < this.n; i++) {
            sumY += this.Ys.get(i);
        }

        return sumY;
    }

    private int calculateSumXY() {
        int sumXY = 0;

        for (int i = 0; i < this.n; i++) {
            sumXY += this.Xs.get(i) * this.Ys.get(i);
        }

        return sumXY;
    }

    private int calculateSumXSquared() {
        int sumXSquared = 0;

        for (int i = 0; i < this.n; i++) {
            sumXSquared += Math.pow(this.Xs.get(i), 2);
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
        double optimalLambda = LinearRegression.LAMBDAS.get(0);

        for (int i = 0; i < LinearRegression.LAMBDAS.size(); i++) {
            System.out.println("----Trying lambda \"" + LinearRegression.LAMBDAS.get(i) + "\"...");

            double current_E_cv = this.calculate_E_cv(this.weights, LinearRegression.LAMBDAS.get(i));

            ArrayList<Double> temporaryResult_E_ins = this.linearRegressionResult.getE_ins();
            temporaryResult_E_ins.add(i, this.calculate_E_in(this.weights, this.Xs, this.Ys));
            this.linearRegressionResult.setE_ins(temporaryResult_E_ins);

            ArrayList<Double> temporaryResult_E_Cvs = this.linearRegressionResult.getE_cvs();
            temporaryResult_E_Cvs.add(i, current_E_cv);
            this.linearRegressionResult.setE_cvs(temporaryResult_E_Cvs);

            if (current_E_cv < smallest_E_cv) {
                System.out.println("----This lambda's E_cv (lambda: \"" + LinearRegression.LAMBDAS.get(i) + "\", E_cv: \"" + current_E_cv + "\") is better than the best lambda's E_cv so far (lambda: \"" + optimalLambda + "\", E_cv: \"" + smallest_E_cv + "\"), reassigning it.");

                smallest_E_cv = current_E_cv;
                optimalLambda = LinearRegression.LAMBDAS.get(i);
            } else {
                System.out.println("----This lambda's E_cv (lambda: \"" + LinearRegression.LAMBDAS.get(i) + "\", E_cv: \"" + current_E_cv + "\") is NOT better than the best lambda's E_cv so far (lambda: \"" + optimalLambda + "\", E_cv: \"" + smallest_E_cv + "\").");
            }
        }

        System.out.println("--Calculated optimal lambda to be \"" + optimalLambda + "\".");
        return optimalLambda;
    }

    private double calculate_E_cv(final ArrayList<Double> weights, final double lambda) {
        System.out.println("------Calculating E_cv for lambda \"" + lambda + "\" ((sum(for each leaveOneOut set: leaveOneOut_E_aug(lambda))) / n)...");

        double E_cv;
        ArrayList<Double> allLeaveOneOut_E_aug = new ArrayList<>();

        for (int i = 0; i < this.n; i++) {
            final ArrayList<Double> leaveOneOutXs = new ArrayList<>();
            final ArrayList<Double> leaveOneOutYs = new ArrayList<>();

            for (int j = 0; j < this.n; j++) {
                if (i != j) {
                    leaveOneOutXs.add(this.Xs.get(j));
                    leaveOneOutYs.add(this.Ys.get(j));
                }
            }

            allLeaveOneOut_E_aug.add(calculate_E_aug(weights, leaveOneOutXs, leaveOneOutYs, lambda));
        }

        Double sumLeaveOneOut_E_aug = 0.0;
        for (final Double leaveOneOut_E_aug : allLeaveOneOut_E_aug) {
            sumLeaveOneOut_E_aug += leaveOneOut_E_aug;
        }

        E_cv = sumLeaveOneOut_E_aug / this.n;
        System.out.println("------Calculated E_cv for lambda \"" + lambda + "\" to be \"" + E_cv + "\".");
        return E_cv;
    }

    private double calculate_E_aug(final ArrayList<Double> weights, final ArrayList<Double> Xs, final ArrayList<Double> Ys, final double lambda) {
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

    private double calculate_E_in(final ArrayList<Double> weights, final ArrayList<Double> Xs, final ArrayList<Double> Ys) {
        System.out.println("----------Calculating E_in ((sum((wTx - y)^2))/n)...");

        double E_in;
        final int n = Xs.size();

        double sumE_in = 0.0;

        for (int i = 0; i < n; i++) {
            final double X_i = Xs.get(i);
            final double Y_i = Ys.get(i);

            sumE_in += Math.pow((this.calculate_wTx(weights, X_i) - Y_i), 2);
        }

        E_in = sumE_in / n;
        System.out.println("----------Calculated E_in to be \"" + E_in + "\".");
        return E_in;
    }

    private double calculate_wTx(final ArrayList<Double> weights, final double X_1) {
        System.out.println("------------Calculating wTx (" + weights.get(0) + " * " + LinearRegression.X_0 + " + " + weights.get(1) + " * " + X_1 + ")...");

        double wTx = weights.get(0) * LinearRegression.X_0 + weights.get(1) * X_1;
        System.out.println("------------Calculated wTx to be \"" + wTx + "\".");
        return wTx;
    }
}

class LinearRegressionResult {
    private ArrayList<Double> Xs;
    private ArrayList<Double> Ys;
    private String regressionLine;
    private ArrayList<Double> lambdas;
    private ArrayList<Double> E_ins = new ArrayList<>();
    private ArrayList<Double> E_cvs = new ArrayList<>();
    private double finalLambda;
    private String regularizedRegressionLine;
    private double final_E_in;

    LinearRegressionResult() {
    }

    void setXs(final ArrayList<Double> xs) {
        Xs = xs;
    }

    void setYs(final ArrayList<Double> ys) {
        Ys = ys;
    }

    void setRegressionLine(final String regressionLine) {
        this.regressionLine = regressionLine;
    }

    void setLambdas(final ArrayList<Double> lambdas) {
        this.lambdas = lambdas;
    }

    ArrayList<Double> getE_ins() {
        return E_ins;
    }

    void setE_ins(final ArrayList<Double> E_ins) {
        this.E_ins = E_ins;
    }

    ArrayList<Double> getE_cvs() {
        return E_cvs;
    }

    void setE_cvs(final ArrayList<Double> E_cvs) {
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
                "Xs=" + this.Xs +
                ", Ys=" + this.Ys +
                ", regressionLine='" + this.regressionLine + '\'' +
                ", lambdas=" + this.lambdas +
                ", E_ins=" + this.E_ins +
                ", E_cvs=" + this.E_cvs +
                ", finalLambda=" + this.finalLambda +
                ", regularizedRegressionLine='" + this.regularizedRegressionLine + '\'' +
                ", final_E_in=" + this.final_E_in +
                '}';
    }

    String toResultString() {
        StringBuilder output = new StringBuilder("\n========================= RESULTS =========================");

        output.append("\n(a) Twelve (X, Y) coordinate pairs: ");
        for (int i = 0; i < this.Xs.size(); i++) {
            output.append("\n    • (").append(this.Xs.get(i)).append(", ").append(this.Ys.get(i)).append(")");
        }

        output.append("\n(b) Original Regression Line:");
        output.append("\n    • \"").append(this.regressionLine).append("\"");

        output.append("\n(c) Four (Lambda, E_in, E_cv) Triplets:");
        output.append("\n    • (").append(this.lambdas.get(0)).append(", ").append(this.E_ins.get(0)).append(", ").append(this.E_cvs.get(0)).append(")");
        output.append("\n    • (").append(this.lambdas.get(1)).append(", ").append(this.E_ins.get(1)).append(", ").append(this.E_cvs.get(1)).append(")");
        output.append("\n    • (").append(this.lambdas.get(2)).append(", ").append(this.E_ins.get(2)).append(", ").append(this.E_cvs.get(2)).append(")");
        output.append("\n    • (").append(this.lambdas.get(3)).append(", ").append(this.E_ins.get(3)).append(", ").append(this.E_cvs.get(3)).append(")");

        output.append("\n(d) Final Lambda:");
        output.append("\n    • ").append(this.finalLambda);

        output.append("\n(e) Regularized Regression Line:");
        output.append("\n    • \"").append(this.regularizedRegressionLine).append("\"");
        output.append("\n    Final E_in:");
        output.append("\n    • ").append(this.final_E_in);

        return output.toString();
    }
}
