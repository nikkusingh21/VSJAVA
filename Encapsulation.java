class Car {
    private String color;

    public void setColor(String color) {
        this.color = color;
    }

    public String getColor() {
        return color;
    }
}

class EncapsulationDemo {
    public static void main(String[] args) {
        Car myCar = new Car();
        myCar.setColor("Green");
        System.out.println(myCar.getColor());
    }
}