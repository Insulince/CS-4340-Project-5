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

        LinearRegression linearRegression = new LinearRegression(Xs, Ys);
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
    private final String regressionAlgorithm;
    private final String regularizedRegressionAlgorithm;

    LinearRegression(final ArrayList<Double> Xs, final ArrayList<Double> Ys) throws Exception {
        if (Xs.size() == Ys.size()) {
            this.Xs = Xs;
            this.Ys = Ys;
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

            this.regressionAlgorithm = "y=" + this.a + (this.b >= 0 ? "*x+" + this.b : "*x-" + -this.b);

            this.regularizedRegressionAlgorithm = this.calculateRegularizedRegression();
        } else {
            throw new Exception("Number of Xs and number of Ys differ, cannot preform linear regression.");
        }
    }

    String getRegressionAlgorithm() {
        return this.regressionAlgorithm;
    }

    String getRegularizedRegressionAlgorithm() {
        return this.regularizedRegressionAlgorithm;
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "Xs=" + this.Xs +
                ", Ys=" + this.Ys +
                ", weights=" + this.weights +
                ", n=" + this.n +
                ", sumX=" + this.sumX +
                ", sumY=" + this.sumY +
                ", sumXY=" + this.sumXY +
                ", sumXSquared=" + this.sumXSquared +
                ", a=" + this.a +
                ", b=" + this.b +
                ", regressionAlgorithm='" + this.regressionAlgorithm + '\'' +
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
        double optimalLambda = this.determineOptimalLambdaViaCrossValidation();

        System.out.println("Optimal Lambda: " + optimalLambda);

        return "y=" + this.a + (this.b >= 0 ? "*x+" + this.b : "*x-" + -this.b);
    }

    private double ridgeRegression(final double lambda) {
        double wTw = 0.0;

        for (final double weight : this.weights) {
            wTw += Math.pow(weight, 2);
        }

        return calculate_E_in(this.weights, this.Xs, this.Ys) + lambda * wTw;
    }

    private double determineOptimalLambdaViaCrossValidation() {
        double smallest_E_cv = Double.MAX_VALUE;
        double mostAppropriateLambda = LinearRegression.LAMBDAS.get(0);

        for (int i = 1; i < LinearRegression.LAMBDAS.size(); i++) {
            double current_E_cv = this.calculate_E_cv(this.weights, LinearRegression.LAMBDAS.get(i));

            if (current_E_cv < smallest_E_cv) {
                smallest_E_cv = current_E_cv;
                mostAppropriateLambda = LinearRegression.LAMBDAS.get(i);
            }
        }

        return mostAppropriateLambda;
    }

    private double calculate_E_cv(final ArrayList<Double> weights, final double lambda) {
        ArrayList<Double> allLeaveOneOut_E_in = new ArrayList<>();

        for (int i = 0; i < this.n; i++) {
            final ArrayList<Double> leaveOneOutXs = new ArrayList<>();
            final ArrayList<Double> leaveOneOutYs = new ArrayList<>();

            for (int j = 0; j < this.n; j++) {
                if (i != j) {
                    leaveOneOutXs.add(this.Xs.get(i));
                    leaveOneOutYs.add(this.Ys.get(i));
                }
            }

            allLeaveOneOut_E_in.add(calculate_E_aug(weights, leaveOneOutXs, leaveOneOutYs, lambda));
        }

        Double sumLeaveOneOut_E_in = 0.0;
        for (int i = 0; i < allLeaveOneOut_E_in.size(); i++) {
            sumLeaveOneOut_E_in += allLeaveOneOut_E_in.get(i);
        }

        return sumLeaveOneOut_E_in / this.n;
    }

    private double calculate_E_aug(final ArrayList<Double> weights, final ArrayList<Double> Xs, final ArrayList<Double> Ys, final double lambda) {
        double wTw = 0.0;

        for (final double weight : weights) {
            wTw += Math.pow(weight, 2);
        }

        return this.calculate_E_in(weights, Xs, Ys) + lambda * wTw;
    }

    private double calculate_E_in(final ArrayList<Double> weights, final ArrayList<Double> Xs, final ArrayList<Double> Ys) {
        final int n = Xs.size();

        double E_in = 0.0;

        for (int i = 0; i < n; i++) {
            final double X_i = Xs.get(i);
            final double Y_i = Ys.get(i);

            E_in += Math.pow((this.calculate_wTx(weights, X_i) - Y_i), 2);
        }

        return E_in / n;
    }

    private double calculate_wTx(final ArrayList<Double> weights, final double X_1) {
        return weights.get(0) * LinearRegression.X_0 + weights.get(1) * X_1;
    }
}
