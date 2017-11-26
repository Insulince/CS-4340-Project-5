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

    public static void main(final String[] args) throws ImproperlySizedSquareMatrixException, ImproperlySizedDeterminantMatrixException, NonInvertibleMatrixException {
//        LinearRegression x;

        ArrayList<ArrayList<Integer>> matrixData = new ArrayList<>();

        ArrayList<Integer> row1 = new ArrayList<>();
        ArrayList<Integer> row2 = new ArrayList<>();
        ArrayList<Integer> row3 = new ArrayList<>();
        ArrayList<Integer> row4 = new ArrayList<>();
        ArrayList<Integer> row5 = new ArrayList<>();
        ArrayList<Integer> row6 = new ArrayList<>();

        row1.add(4);
        row1.add(5);
        row1.add(6);
        row1.add(1);
        row1.add(4);
        row1.add(3);

        row2.add(8);
        row2.add(4);
        row2.add(5);
        row2.add(6);
        row2.add(1);
        row2.add(2);

        row3.add(9);
        row3.add(7);
        row3.add(5);
        row3.add(4);
        row3.add(6);
        row3.add(8);

        row4.add(3);
        row4.add(1);
        row4.add(6);
        row4.add(9);
        row4.add(4);
        row4.add(2);

        row5.add(5);
        row5.add(4);
        row5.add(6);
        row5.add(8);
        row5.add(3);
        row5.add(2);

        row6.add(5);
        row6.add(7);
        row6.add(8);
        row6.add(6);
        row6.add(1);
        row6.add(5);

        matrixData.add(row1);
        matrixData.add(row2);
        matrixData.add(row3);
        matrixData.add(row4);
        matrixData.add(row5);
        matrixData.add(row6);

        InvertibleMatrix invertibleMatrix = new InvertibleMatrix(matrixData);

        System.out.println("Determinant is: " + invertibleMatrix.calculateDeterminant());
    }

    @Override
    public String toString() {
        return "Project5{}";
    }
}

class Matrix {
    private final ArrayList<ArrayList<Integer>> data;

    Matrix(final ArrayList<ArrayList<Integer>> data) {
        this.data = data;
    }

    ArrayList<ArrayList<Integer>> getData() {
        return this.data;
    }

    @Override
    public String toString() {
        return "Matrix{" +
                "data=" + this.data +
                '}';
    }

    public ArrayList<Integer> getRow(final int row) {
        return this.data.get(row);
    }

    int getCell(final int row, final int column) {
        return this.data.get(row).get(column);
    }

    public Matrix transpose() {
        final ArrayList<ArrayList<Integer>> transposedData = new ArrayList<>();

        int maximumColumnsEncountered = 0;
        for (ArrayList<Integer> datum : this.data) {
            if (datum.size() > maximumColumnsEncountered) {
                maximumColumnsEncountered = datum.size();
            }
        }

        for (int i = 0; i < maximumColumnsEncountered; i++) {
            transposedData.add(new ArrayList<>());
        }

        for (int i = 0; i < this.data.size(); i++) {
            for (int j = 0; j < this.data.get(i).size(); j++) {
                transposedData.get(j).add(i, this.data.get(i).get(j));
            }

            transposedData.add(new ArrayList<>());
        }

        return new Matrix(transposedData);
    }
}

class SquareMatrix extends Matrix {
    private final int size;

    SquareMatrix(final ArrayList<ArrayList<Integer>> data) throws ImproperlySizedSquareMatrixException {
        super(data);

        final int EXPECTED_QUANTITY_CELLS_IN_EACH_ROW = this.getData().get(0).size();
        for (int i = 0; i < this.getData().size(); i++) {
            if (this.getData().get(i).size() != EXPECTED_QUANTITY_CELLS_IN_EACH_ROW) {
                throw new ImproperlySizedSquareMatrixException(EXPECTED_QUANTITY_CELLS_IN_EACH_ROW, this.getData().get(i).size(), i);
            }
        }

        this.size = EXPECTED_QUANTITY_CELLS_IN_EACH_ROW;
    }

    public int getSize() {
        return this.size;
    }

    @Override
    public String toString() {
        return "SquareMatrix{" +
                "size=" + this.size +
                '}';
    }

    int calculateDeterminant() throws ImproperlySizedDeterminantMatrixException, ImproperlySizedSquareMatrixException {
        int determinant = 0;

        if (this.size > 2) {
            for (int topMostValueIndex = 0; topMostValueIndex < this.size; topMostValueIndex++) {
                int topMostValueForThisColumn = this.getCell(0, topMostValueIndex);

                if (topMostValueIndex % 2 != 0) {
                    topMostValueForThisColumn *= -1;
                }

                ArrayList<ArrayList<Integer>> newMatrixData = new ArrayList<>();

                for (int rowIndexToBeProcessed = 1; rowIndexToBeProcessed < this.size; rowIndexToBeProcessed++) {
                    newMatrixData.add(new ArrayList<>());

                    for (int columnIndexToBeProcessed = 0; columnIndexToBeProcessed < this.size; columnIndexToBeProcessed++) {
                        if (columnIndexToBeProcessed != topMostValueIndex) {
                            newMatrixData.get(rowIndexToBeProcessed - 1).add(this.getCell(rowIndexToBeProcessed, columnIndexToBeProcessed));
                        }
                    }
                }

                SquareMatrix newSquareMatrix = new SquareMatrix(newMatrixData);

                determinant += topMostValueForThisColumn * newSquareMatrix.calculateDeterminant();
            }
        } else if (this.size == 2) {
            determinant = this.getCell(0, 0) * this.getCell(1, 1) - (this.getCell(0, 1) * this.getCell(1, 0));
        } else if (this.size == 1) {
            determinant = this.getCell(0, 0);
        } else {
            throw new ImproperlySizedDeterminantMatrixException(this.size);
        }

        return determinant;
    }
}

class ImproperlySizedSquareMatrixException extends Exception {
    ImproperlySizedSquareMatrixException(final int expectedQuantityColumns, final int encounteredQuantityColumns, final int encounteredOnRow) {
        super("Improperly sized Square Matrix encountered. Expected number of columns: \"" + expectedQuantityColumns + "\" (based on first row in matrix), encountered number of columns: \"" + encounteredQuantityColumns + "\" on row \"" + encounteredOnRow + "\".");
    }
}

class ImproperlySizedDeterminantMatrixException extends Exception {
    ImproperlySizedDeterminantMatrixException(int encounteredSize) {
        super("Improperly sized determinant matrix encountered. Expected size > 0, encountered size \"" + encounteredSize + "\".");
    }
}

class InvertibleMatrix extends SquareMatrix {
    InvertibleMatrix(final ArrayList<ArrayList<Integer>> data) throws NonInvertibleMatrixException, ImproperlySizedSquareMatrixException, ImproperlySizedDeterminantMatrixException {
        super(data);

        if (this.calculateDeterminant() == 0) {
            throw new NonInvertibleMatrixException();
        }
    }

//    public InvertibleMatrix invert() {
//
//    }
}

class NonInvertibleMatrixException extends Exception {
    NonInvertibleMatrixException() {
        super("Provided matrix has a determinant of zero therefore no inverse exists!");
    }
}

class LinearRegression {
    private final Data data;
    private final ArrayList<Weight> weights;
    private final int n;
    private final int d;

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

        this.d = EXPECTED_QUANTITY_FEATURES_IN_EACH_DATUM + 1; // + 1 for bias, which is not added yet (added in next segment).

        this.weights.add(Weight.BIAS_WEIGHT);

        for (int i = 1; i < d; i++) {
            this.weights.add(new Weight());
        }
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

    @Override
    public String toString() {
        return "LinearRegression{" +
                "data=" + this.data +
                ", weights=" + this.weights +
                '}';
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

    private double E_in() {
        double E_in = 0.0;

        for (int i = 0; i < this.n; i++) {
            E_in += Math.pow((this.h(this.data.getFeatures().get(i)) - this.data.getOutcomes().get(i).getValue()), 2);
        }

        E_in /= this.n;

        return E_in;
    }

    private double h(Features x) {
        double wTx = 0.0;

        for (int j = 0; j < this.d; j++) {
            wTx += this.weights.get(j).getValue() * x.getFeatures().get(j).getValue();
        }

        return wTx;
    }
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

    private Feature(final int value) {
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
