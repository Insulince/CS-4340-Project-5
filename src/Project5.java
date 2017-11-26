import java.util.ArrayList;

/*                                                                                   [BIAS_WEIGHT,  Weight,  ...{d times}..., Weight ]
                                                                 ,-              -, ,-                                                -,
                                                                 |     Outcome,   | |[BIAS_FEATURE, Feature, ...{d times}..., Feature],|
                                                                 |     Outcome,   | |[BIAS_FEATURE, Feature, ...{d times}..., Feature],|
                                                                 |     Outcome,   | |[BIAS_FEATURE, Feature, ...{d times}..., Feature],|
                                                                 |...{n times}...,| |                 ...{n times}...,                 |
                                                                 |     Outcome    | |[BIAS_FEATURE, Feature, ...{d times}..., Feature] |
                                                                 '-              -' '-                                                -'                                                                               */

public class Project5 {
    public Project5() {
    }

    public static void main(final String[] args) throws ImproperlySizedDatumFeaturesException {
        Feature feature1_1 = new Feature(1);
        ArrayList<Feature> features1List = new ArrayList<>();
        features1List.add(feature1_1);
        Features features1 = new Features(features1List);

        Feature feature2_1 = new Feature(2);
        ArrayList<Feature> features2List = new ArrayList<>();
        features2List.add(feature2_1);
        Features features2 = new Features(features2List);

        Feature feature3_1 = new Feature(3);
        ArrayList<Feature> features3List = new ArrayList<>();
        features3List.add(feature3_1);
        Features features3 = new Features(features3List);

        Feature feature4_1 = new Feature(4);
        ArrayList<Feature> features4List = new ArrayList<>();
        features4List.add(feature4_1);
        Features features4 = new Features(features4List);

        Feature feature5_1 = new Feature(5);
        ArrayList<Feature> features5List = new ArrayList<>();
        features5List.add(feature5_1);
        Features features5 = new Features(features5List);

        Feature feature6_1 = new Feature(6);
        ArrayList<Feature> features6List = new ArrayList<>();
        features6List.add(feature6_1);
        Features features6 = new Features(features6List);

        Feature feature7_1 = new Feature(7);
        ArrayList<Feature> features7List = new ArrayList<>();
        features7List.add(feature7_1);
        Features features7 = new Features(features7List);

        Feature feature8_1 = new Feature(8);
        ArrayList<Feature> features8List = new ArrayList<>();
        features8List.add(feature8_1);
        Features features8 = new Features(features8List);

        ArrayList<Features> features = new ArrayList<>();
        features.add(features1);
        features.add(features2);
        features.add(features3);
        features.add(features4);
        features.add(features5);
        features.add(features6);
        features.add(features7);
        features.add(features8);

        Outcome outcome1 = new Outcome(0);
        Outcome outcome2 = new Outcome(1);
        Outcome outcome3 = new Outcome(0);
        Outcome outcome4 = new Outcome(1);
        Outcome outcome5 = new Outcome(0);
        Outcome outcome6 = new Outcome(1);
        Outcome outcome7 = new Outcome(1);
        Outcome outcome8 = new Outcome(1);
        ArrayList<Outcome> outcomes = new ArrayList<>();
        outcomes.add(outcome1);
        outcomes.add(outcome2);
        outcomes.add(outcome3);
        outcomes.add(outcome4);
        outcomes.add(outcome5);
        outcomes.add(outcome6);
        outcomes.add(outcome7);
        outcomes.add(outcome8);

        Data data = new Data(features, outcomes);
        LinearRegression linearRegression = new LinearRegression(data);

        System.out.println(linearRegression);
    }

    @Override
    public String toString() {
        return "Project5{}";
    }
}

class LinearRegression {
    private final Data data;
    private final ArrayList<Weight> weights;
    private final int n;
    private final int d;
    private final int sumX;
    private final int sumY;
    private final int sumXY;
    private final int sumXSquared;
    private final double a;
    private final double b;
    private final String regressionAlgorithm;

    LinearRegression(final Data data) throws ImproperlySizedDatumFeaturesException {
        this.data = data;
        this.weights = new ArrayList<>();
        this.n = this.data.getFeatures().size();

        final int EXPECTED_QUANTITY_FEATURES_IN_EACH_DATUM = this.data.getFeatures().get(0).getFeatures().size();
        for (int i = 1; i < this.data.getFeatures().size(); i++) {
            if (this.data.getFeatures().get(i).getFeatures().size() != EXPECTED_QUANTITY_FEATURES_IN_EACH_DATUM) {
                throw new ImproperlySizedDatumFeaturesException(EXPECTED_QUANTITY_FEATURES_IN_EACH_DATUM, this.data.getFeatures().get(i).getFeatures().size());
            }
        }

        this.d = EXPECTED_QUANTITY_FEATURES_IN_EACH_DATUM;

        this.weights.add(0, Weight.BIAS_WEIGHT);

        for (int i = 1; i < d; i++) {
            this.weights.add(new Weight());
        }

        this.sumX = this.calculateSumX();
        this.sumY = this.calculateSumY();
        this.sumXY = this.calculateSumXY();
        this.sumXSquared = this.calculateSumXSquared();

        this.a = this.calculateA();
        this.b = this.calculateB();

        this.regressionAlgorithm = "y = " + this.a + (this.b >= 0 ? " * x + " + this.b : "x - " + -this.b);
    }

    Data getData() {
        return this.data;
    }

    ArrayList<Weight> getWeights() {
        return this.weights;
    }

    int getN() {
        return this.n;
    }

    int getD() {
        return this.d;
    }

    public int getSumX() {
        return this.sumX;
    }

    public int getSumY() {
        return this.sumY;
    }

    public int getSumXY() {
        return this.sumXY;
    }

    public int getSumXSquared() {
        return this.sumXSquared;
    }

    public double getA() {
        return this.a;
    }

    public double getB() {
        return this.b;
    }

    public String getRegressionAlgorithm() {
        return this.regressionAlgorithm;
    }

    @Override
    public String toString() {
        return "LinearRegression{" +
                "data=" + this.data +
                ", weights=" + this.weights +
                ", n=" + this.n +
                ", d=" + this.d +
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
            sumX += this.data.getFeatures().get(i).getFeatures().get(1).getValue();
        }

        return sumX;
    }

    private int calculateSumY() {
        int sumY = 0;

        for (int i = 0; i < this.n; i++) {
            sumY += this.data.getOutcomes().get(i).getValue();
        }

        return sumY;
    }

    private int calculateSumXY() {
        int sumXY = 0;

        for (int i = 0; i < this.n; i++) {
            sumXY += this.data.getFeatures().get(i).getFeatures().get(1).getValue() * this.data.getOutcomes().get(i).getValue();
        }

        return sumXY;
    }

    private int calculateSumXSquared() {
        int sumXSquared = 0;

        for (int i = 0; i < this.n; i++) {
            sumXSquared += Math.pow(this.data.getFeatures().get(i).getFeatures().get(1).getValue(), 2);
        }

        return sumXSquared;
    }


    private double calculateA() {
        return (this.n * this.sumXY - this.sumX * this.sumY) / (this.n * this.sumXSquared - Math.pow(this.sumX, 2));
    }

    private double calculateB() {
        return (this.sumY - this.a * this.sumX) / this.n;
    }

//    public void train() {
//        //(X*w - Y)^2
//        //(X*w*X*w - 2*Y*X*w - Y^2)
//
//        ArrayList<ArrayList<Integer>> xTx = new ArrayList<>();
//
//        for (int i = 0; i < this.n; i++) {
//            this.data.getFeatures().get(i);
//
//            xTx.add(new ArrayList<>());
//
//            for (int j = 0; j < this.d; j++) {
//                for (int k = 0; k < this.d; k++) {
//                    xTx.get(i).add(this.data.getFeatures().get(i).getFeatures().get(j).getValue() * this.data.getFeatures().get(i).getFeatures().get(k).getValue());
//                }
//            }
//        }
//
//        for (int i = 0; i < this.d; i++) {
//            for (int j = 0; j < this.d; j++) {
//                //Invert
//            }
//        }
//    }

//    private double E_in() {
//        double E_in = 0.0;
//
//        for (int i = 0; i < this.n; i++) {
//            E_in += Math.pow((this.h(this.data.getFeatures().get(i)) - this.data.getOutcomes().get(i).getValue()), 2);
//        }
//
//        E_in /= this.n;
//
//        return E_in;
//    }
//
//    private double h(Features x) {
//        double wTx = 0.0;
//
//        for (int j = 0; j < this.d; j++) {
//            wTx += this.weights.get(j).getValue() * x.getFeatures().get(j).getValue();
//        }
//
//        return wTx;
//    }
}

class Data {
    private final ArrayList<Features> features;
    private final ArrayList<Outcome> outcomes;

    Data(final ArrayList<Features> features, final ArrayList<Outcome> outcomes) {
        this.features = features;
        this.outcomes = outcomes;
    }

    ArrayList<Features> getFeatures() {
        return this.features;
    }

    ArrayList<Outcome> getOutcomes() {
        return this.outcomes;
    }

    @Override
    public String toString() {
        return "Data{" +
                "features=" + this.features +
                ", outcomes=" + this.outcomes +
                '}';
    }
}

class Features {
    private final ArrayList<Feature> features;

    Features(final ArrayList<Feature> features) {
        this.features = features;

        this.features.add(0, Feature.BIAS_FEATURE);
    }

    ArrayList<Feature> getFeatures() {
        return this.features;
    }

    @Override
    public String toString() {
        return "Features{" +
                "features=" + this.features +
                '}';
    }
}

class Feature {
    static final Feature BIAS_FEATURE = new Feature(1);

    private final int value;

    Feature(final int value) {
        this.value = value;
    }

    int getValue() {
        return this.value;
    }

    @Override
    public String toString() {
        return "Feature{" +
                "value=" + this.value +
                '}';
    }
}

class Outcome {
    private final int value;

    Outcome(final int value) {
        this.value = value;
    }

    int getValue() {
        return this.value;
    }

    @Override
    public String toString() {
        return "Outcome{" +
                "value=" + this.value +
                '}';
    }
}

class Weight {
    static final Weight BIAS_WEIGHT = new Weight();

    private double value;

    Weight() {
        this(Math.random());
    }

    private Weight(final double value) {
        this.value = value;
    }

    double getValue() {
        return this.value;
    }

    void setValue(final double value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "Weight{" +
                "value=" + this.value +
                '}';
    }
}

class ImproperlySizedDatumFeaturesException extends Exception {
    ImproperlySizedDatumFeaturesException(final int expectedQuantityFeatures, final int encounteredQuantityFeatures) {
        super("Improperly sized Data Features encountered. Expected size \"" + expectedQuantityFeatures + "\" (based on first Data in data set), encountered size: \"" + encounteredQuantityFeatures + "\".");
    }
}

//class Matrix {
//    private final ArrayList<ArrayList<Integer>> data;
//
//    Matrix(final ArrayList<ArrayList<Integer>> data) {
//        this.data = data;
//    }
//
//    ArrayList<ArrayList<Integer>> getData() {
//        return this.data;
//    }
//
//    @Override
//    public String toString() {
//        return "Matrix{" +
//                "data=" + this.data +
//                '}';
//    }
//
//    public ArrayList<Integer> getRow(final int row) {
//        return this.data.get(row);
//    }
//
//    int getCell(final int row, final int column) {
//        return this.data.get(row).get(column);
//    }
//
//    public Matrix transpose() {
//        final ArrayList<ArrayList<Integer>> transposedData = new ArrayList<>();
//
//        int maximumColumnsEncountered = 0;
//        for (ArrayList<Integer> datum : this.data) {
//            if (datum.size() > maximumColumnsEncountered) {
//                maximumColumnsEncountered = datum.size();
//            }
//        }
//
//        for (int i = 0; i < maximumColumnsEncountered; i++) {
//            transposedData.add(new ArrayList<>());
//        }
//
//        for (int i = 0; i < this.data.size(); i++) {
//            for (int j = 0; j < this.data.get(i).size(); j++) {
//                transposedData.get(j).add(i, this.data.get(i).get(j));
//            }
//
//            transposedData.add(new ArrayList<>());
//        }
//
//        return new Matrix(transposedData);
//    }
//}
//
//class SquareMatrix extends Matrix {
//    private final int size;
//
//    SquareMatrix(final ArrayList<ArrayList<Integer>> data) throws ImproperlySizedSquareMatrixException {
//        super(data);
//
//        final int EXPECTED_QUANTITY_CELLS_IN_EACH_ROW = this.getData().get(0).size();
//        for (int i = 0; i < this.getData().size(); i++) {
//            if (this.getData().get(i).size() != EXPECTED_QUANTITY_CELLS_IN_EACH_ROW) {
//                throw new ImproperlySizedSquareMatrixException(EXPECTED_QUANTITY_CELLS_IN_EACH_ROW, this.getData().get(i).size(), i);
//            }
//        }
//
//        this.size = EXPECTED_QUANTITY_CELLS_IN_EACH_ROW;
//    }
//
//    public int getSize() {
//        return this.size;
//    }
//
//    @Override
//    public String toString() {
//        return "SquareMatrix{" +
//                "size=" + this.size +
//                '}';
//    }
//
//    int calculateDeterminant() throws ImproperlySizedDeterminantMatrixException, ImproperlySizedSquareMatrixException {
//        int determinant = 0;
//
//        if (this.size > 2) {
//            for (int topMostValueIndex = 0; topMostValueIndex < this.size; topMostValueIndex++) {
//                int topMostValueForThisColumn = this.getCell(0, topMostValueIndex);
//
//                if (topMostValueIndex % 2 != 0) {
//                    topMostValueForThisColumn *= -1;
//                }
//
//                ArrayList<ArrayList<Integer>> newMatrixData = new ArrayList<>();
//
//                for (int rowIndexToBeProcessed = 1; rowIndexToBeProcessed < this.size; rowIndexToBeProcessed++) {
//                    newMatrixData.add(new ArrayList<>());
//
//                    for (int columnIndexToBeProcessed = 0; columnIndexToBeProcessed < this.size; columnIndexToBeProcessed++) {
//                        if (columnIndexToBeProcessed != topMostValueIndex) {
//                            newMatrixData.get(rowIndexToBeProcessed - 1).add(this.getCell(rowIndexToBeProcessed, columnIndexToBeProcessed));
//                        }
//                    }
//                }
//
//                SquareMatrix newSquareMatrix = new SquareMatrix(newMatrixData);
//
//                determinant += topMostValueForThisColumn * newSquareMatrix.calculateDeterminant();
//            }
//        } else if (this.size == 2) {
//            determinant = this.getCell(0, 0) * this.getCell(1, 1) - (this.getCell(0, 1) * this.getCell(1, 0));
//        } else if (this.size == 1) {
//            determinant = this.getCell(0, 0);
//        } else {
//            throw new ImproperlySizedDeterminantMatrixException(this.size);
//        }
//
//        return determinant;
//    }
//}
//
//class ImproperlySizedSquareMatrixException extends Exception {
//    ImproperlySizedSquareMatrixException(final int expectedQuantityColumns, final int encounteredQuantityColumns, final int encounteredOnRow) {
//        super("Improperly sized Square Matrix encountered. Expected number of columns: \"" + expectedQuantityColumns + "\" (based on first row in matrix), encountered number of columns: \"" + encounteredQuantityColumns + "\" on row \"" + encounteredOnRow + "\".");
//    }
//}
//
//class ImproperlySizedDeterminantMatrixException extends Exception {
//    ImproperlySizedDeterminantMatrixException(int encounteredSize) {
//        super("Improperly sized determinant matrix encountered. Expected size > 0, encountered size \"" + encounteredSize + "\".");
//    }
//}
//
//class InvertibleMatrix extends SquareMatrix {
//    InvertibleMatrix(final ArrayList<ArrayList<Integer>> data) throws NonInvertibleMatrixException, ImproperlySizedSquareMatrixException, ImproperlySizedDeterminantMatrixException {
//        super(data);
//
//        if (this.calculateDeterminant() == 0) {
//            throw new NonInvertibleMatrixException();
//        }
//    }
//
////    public InvertibleMatrix invert() {
////
////    }
//}
//
//class NonInvertibleMatrixException extends Exception {
//    NonInvertibleMatrixException() {
//        super("Provided matrix has a determinant of zero therefore no inverse exists!");
//    }
//}

