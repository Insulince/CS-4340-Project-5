import java.util.ArrayList;

public class Project5 {
    private static final int QUANTITY_RANDOM_DOUBLES = 12;
    private static final double RANDOM_DOUBLE_LOWER_BOUND = -2;
    private static final double RANDOM_DOUBLE_UPPER_BOUND = 10;

    public static void main(final String[] args) {
        final ArrayList<Double> features = new ArrayList<>();
        for (int i = 0; i < Project5.QUANTITY_RANDOM_DOUBLES; i++) {
            features.add(Project5.randomDouble(Project5.RANDOM_DOUBLE_LOWER_BOUND, Project5.RANDOM_DOUBLE_UPPER_BOUND));
        }

        final ArrayList<Double> outcomes = new ArrayList<>();
        for (final Double feature : features) {
            outcomes.add(randomSampleFunction(feature));
        }

        LinearRegression linearRegression = new LinearRegression(features, outcomes);

        System.out.println(linearRegression.getRegressionAlgorithm());
    }

    private static double randomDouble(final double lowerBoundInclusive, final double upperBoundInclusive) {
        return lowerBoundInclusive + (Math.random() * ((upperBoundInclusive - lowerBoundInclusive) + 1));
    }

    private static double randomSampleFunction(final double feature) {
        return Math.pow(feature, 2) + 10;
    }
}

class LinearRegression {
    private final ArrayList<Double> features;
    private final ArrayList<Double> outcomes;
    private final ArrayList<Double> weights;
    private final int n;
    private final int sumX;
    private final int sumY;
    private final int sumXY;
    private final int sumXSquared;
    private final double a;
    private final double b;
    private final String regressionAlgorithm;

    LinearRegression(final ArrayList<Double> features, final ArrayList<Double> outcomes) {
        this.features = features;
        this.outcomes = outcomes;
        this.weights = new ArrayList<>();
        this.n = this.features.size();

        this.weights.add(Math.random()); // Bias Weight
        this.weights.add(Math.random()); // Feature 1 Weight

        this.sumX = this.calculateSumX();
        this.sumY = this.calculateSumY();
        this.sumXY = this.calculateSumXY();
        this.sumXSquared = this.calculateSumXSquared();

        this.a = this.calculateA();
        this.b = this.calculateB();

        this.regressionAlgorithm = "y=" + this.a + (this.b >= 0 ? "*x+" + this.b : "*x-" + -this.b);
    }

    String getRegressionAlgorithm() {
        return this.regressionAlgorithm;
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "features=" + this.features +
                ", outcomes=" + this.outcomes +
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
            sumX += this.features.get(i);
        }

        return sumX;
    }

    private int calculateSumY() {
        int sumY = 0;

        for (int i = 0; i < this.n; i++) {
            sumY += this.outcomes.get(i);
        }

        return sumY;
    }

    private int calculateSumXY() {
        int sumXY = 0;

        for (int i = 0; i < this.n; i++) {
            sumXY += this.features.get(i) * this.outcomes.get(i);
        }

        return sumXY;
    }

    private int calculateSumXSquared() {
        int sumXSquared = 0;

        for (int i = 0; i < this.n; i++) {
            sumXSquared += Math.pow(this.features.get(i), 2);
        }

        return sumXSquared;
    }

    private double calculateA() {
        return (this.n * this.sumXY - this.sumX * this.sumY) / (this.n * this.sumXSquared - Math.pow(this.sumX, 2));
    }

    private double calculateB() {
        return (this.sumY - this.a * this.sumX) / this.n;
    }
}
