abstract class Vehicle {
    abstract void start();
}

class Car extends Vehicle {
    void start() {
        System.out.println("Car started");
    }
}

class AbstractionDemo {
    public static void main(String[] args) {
        Vehicle v = new Car();
        v.start();
    }
}