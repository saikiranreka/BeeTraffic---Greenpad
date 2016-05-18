package main;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import opencv.Imageoperations;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import config.Configuration;

public class BeeTraffic {

	static {
		File file = new File("lib/libopencv_java244.so");
		System.load(file.getAbsolutePath());
	}

	public static void main(String[] args) {

		Imageoperations imageop = new Imageoperations();
		String target_path = Configuration.OUTPUT;
		String input_path = Configuration.INPUT;
		Mat[] images = imageop.readImagesfromFolder(input_path);
		String[] names = imageop.getNames();

		for (int i = 0; i < images.length; i++) {
			Mat image = images[i];

			// Crop the part of image where landing pad possibly be.
			image = image.submat(Configuration.ROW_START,
					Configuration.ROW_END, Configuration.COL_START,
					Configuration.COL_END);
			Mat copy_image = image.clone();

			// Adjust Brightness
			image = adjustBrightness(image);

			// convert to HSV
			Mat hsv = new Mat();
			Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV);

			// Identify Green color
			Mat dst2 = identifyGreen(hsv);

			// for debugging
			imageop.writeImage(target_path + "/" + "green_" + names[i], dst2);

			// Find the green Landing pad and get the ROI
			Rect rect = findLandingPad(dst2);

			image = image.submat(rect);
			copy_image = image.clone();
			Mat source = copy_image.clone();
			Mat grayImage = copy_image.clone();

			// Remove green background from the image
			source = removeBackground(image);

			imageop.writeImage(target_path + "/" + "withoutbackground_"
					+ names[i], source);

			// Remove noise
			Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
					new Size(2, 2));
			for (int j = 0; j < 3; j++)
				Imgproc.dilate(source, source, element);

			for (int j = 0; j < 5; j++)
				Imgproc.erode(source, source, element);
			// for debugging
			imageop.writeImage(target_path + "/" + "erode_" + names[i], source);

			Imgproc.cvtColor(source, grayImage, Imgproc.COLOR_RGB2GRAY);

			// find the total area of bees in the image
			double totalArea = findBeeArea(grayImage, copy_image, target_path
					+ "/res_" + names[i]);

			// System.out.println(names[i]+","+Math.round(totalArea
			// / Configuration.AVG_BEE_AREA));
			System.out.println(Math.round(totalArea
					/ Configuration.AVG_BEE_AREA));
		}

	}

	/**
	 * @param image
	 *            - Mat. Image in Mat to calculate the brightness
	 * @return brightness - double. Brightness of the image
	 */
	private static double getBrightness(Mat image) {
		Mat lum = new Mat();
		List<Mat> color = new ArrayList<Mat>();
		Core.split(image, color);

		Core.multiply(color.get(0), new Scalar(0.299), color.get(0));
		Core.multiply(color.get(1), new Scalar(0.587), color.get(1));
		Core.multiply(color.get(2), new Scalar(0.114), color.get(2));

		Core.add(color.get(0), color.get(1), lum);
		Core.add(lum, color.get(2), lum);

		Scalar summ = Core.sumElems(lum);

		double brightness = summ.val[0] / (image.rows() * image.cols() * 2);
		return brightness;
	}

	private static Mat RotateImage(Mat rotImg, double theta) {
		double angleToRot = theta;

		Mat rotatedImage = new Mat();
		if (angleToRot >= 92 && angleToRot <= 93) {
			Core.transpose(rotImg, rotatedImage);
		} else {
			org.opencv.core.Point center = new org.opencv.core.Point(
					rotImg.cols() / 2, rotImg.rows() / 2);
			Mat rotImage = Imgproc.getRotationMatrix2D(center, angleToRot, 1.0);

			Imgproc.warpAffine(rotImg, rotatedImage, rotImage, rotImg.size());
		}

		return rotatedImage;

	}

	/**
	 * @param image
	 *            - Mat to adjust the brightness
	 * @return image - Mat with adjusted brightness (80 <= brightness <=90)
	 */
	private static Mat adjustBrightness(Mat image) {
		double alpha = 1.5;
		double beta = 4;
		double brig = getBrightness(image);
		// System.out.println(brig);
		if (brig < 40) {
			beta = 91 - brig;
			image.convertTo(image, -1, alpha, beta);
		}
		if (brig > 90) {

			image.convertTo(image, -1, 1, -40);
		}
		// brig = getBrightness(image);
		// System.out.println(brig);
		return image;
	}

	/**
	 * @param hsv
	 *            - Mat. Image convert to HSV
	 * @return dst1 - Mat. Image with white pixels representing red color.
	 */
	private static Mat identifyRed(Mat hsv) {
		Mat dst1 = new Mat();
		Mat dst = new Mat();

		Core.inRange(hsv, new Scalar(0, 80, 80), new Scalar(10, 255, 255), dst);
		Core.inRange(hsv, new Scalar(160, 80, 80), new Scalar(179, 255, 255),
				dst1);

		Core.add(dst, dst1, dst1);

		// erode and dialate
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,
				new Size(2, 2));
		for (int j = 0; j < 5; j++)
			Imgproc.erode(dst1, dst1, element);
		for (int j = 0; j < 2; j++)
			Imgproc.dilate(dst1, dst1, element);

		return dst1;
	}

	/**
	 * @param hsv
	 *            - Mat. Image converted to HSV
	 * @return dst1 - Mat. Image with white pixels representing green color.
	 */
	private static Mat identifyGreen(Mat hsv) {
		Mat dst2 = new Mat();

		Core.inRange(hsv, new Scalar(35, 50, 50), new Scalar(90, 255, 255),
				dst2);
		// Core.inRange(hsv, new Scalar(35, 45, 50), new Scalar(90, 255, 255),
		// dst2);

		// erode and dialate
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
				new Size(2, 2));

		for (int j = 0; j < 6; j++)
			Imgproc.erode(dst2, dst2, element);
		for (int j = 0; j < 3; j++)
			Imgproc.dilate(dst2, dst2, element);
		return dst2;
	}

	/**
	 * @param dst2
	 *            - Mat, Green color identified in the image
	 * @return Rect - Rect ROI
	 */
	private static Rect findLandingPad(Mat dst2) {
		ArrayList<MatOfPoint> points = new ArrayList<MatOfPoint>();
		Imgproc.findContours(dst2, points, new Mat(), Imgproc.RETR_LIST,
				Imgproc.CHAIN_APPROX_NONE);
		Collections.sort(points, new Comparator<MatOfPoint>() {

			@Override
			public int compare(MatOfPoint o1, MatOfPoint o2) {
				Rect rect1 = Imgproc.boundingRect(o1);
				Rect rect2 = Imgproc.boundingRect(o2);
				if (rect1.y >= rect2.y) {
					return 1;
				} else {
					return -1;
				}
			}
		});
		double avg = 0;
		int count = 0;
		// System.out.println("size = "+points.size());
		for (MatOfPoint contour : points) {

			Rect rect = Imgproc.boundingRect(contour);
			double area = Imgproc.contourArea(contour);
			// System.out.println("area "+area);
			if (area > 3900) {
				avg = rect.y;
				count = 1;
				break;
			}
			if (area > 20) {
				count++;
				avg += rect.y;
			}
		}

		avg = avg / count;
		MatOfPoint allPoints = new MatOfPoint();
		for (MatOfPoint contour : points) {
			Rect rect = Imgproc.boundingRect(contour);

			if (rect.y > avg - 23 && rect.y < avg + 23) {

				allPoints.push_back(contour);

			}

		}
		Rect rect = Imgproc.boundingRect(allPoints);

		// crop the landing area from image
		if (rect.area() > 9000) {
			avg = 0;
			int change = 20;

			for (MatOfPoint contour : points) {
				Rect temprect = Imgproc.boundingRect(contour);
				double area = Imgproc.contourArea(contour);

				if (area > 4000) {
					avg = temprect.y;
					count = 1;
					change = 7;
					break;
				}
				if (area > 20) {
					count++;
					avg += temprect.y;
				}
			}

			avg = avg / count;
			allPoints = new MatOfPoint();
			for (MatOfPoint contour : points) {

				double area = Imgproc.contourArea(contour);
				Rect temprect = Imgproc.boundingRect(contour);

				if (temprect.y > avg - change && temprect.y < avg + change
						&& area > 63) {
					allPoints.push_back(contour);
				}
			}
			rect = Imgproc.boundingRect(allPoints);
		}
		return rect;
	}

	/**
	 * @param image
	 *            - Mat representing the white landing pad
	 * @return source - Mat with white background removed and black pixels
	 *         representing bees
	 */
	private static Mat removeBackground(Mat image) {
		Mat source = image.clone();
		for (int row = 0; row < image.rows(); row++) {
			for (int col = 0; col < image.cols(); col++) {
				double[] pixel = image.get(row, col);
				if (row <= 3 || col <= 3 || row >= image.rows() - 7
						|| col >= image.cols() - 2) {
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				} else if ((pixel[0] < pixel[1] && pixel[2] < pixel[1])
						|| (2 * pixel[0] <= pixel[1]
								&& 2 * pixel[0] <= pixel[2] && Math
								.abs(pixel[1] - pixel[2]) < 20)) {
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;

				} else if (pixel[0] >= 230 && pixel[1] >= 230
						&& pixel[2] >= 230) {
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;

				} else {
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;

				}
				source.put(row, col, pixel);
			}
		}
		return source;
	}

	/**
	 * @param source
	 *            - Mat. Image after removing green background, noise and
	 *            converting to Gray
	 * @param image
	 *            - Mat. Original Image with landing pad to draw contours
	 * @param name
	 *            - String. Name of the image
	 * @return area - double. Total area of the bees in the image
	 */
	private static double findBeeArea(Mat grayImage, Mat image, String name) {
		Imageoperations imageop = new Imageoperations();
		Mat bImage = image.clone();
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		List<MatOfPoint> refinedContours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(grayImage, contours, new Mat(),
				Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		Imgproc.drawContours(bImage, contours, -1, new Scalar(0, 0, 255));
		double totalArea = 0;
		for (MatOfPoint contour : contours) {
			double area = Imgproc.contourArea(contour);
			if (area > 20 && area < 3000) {
				totalArea += area;
				refinedContours.add(contour);

			}
		}
		Imgproc.drawContours(image, refinedContours, -1, new Scalar(0, 0, 255));
		imageop.writeImage(name, image);
		return totalArea;
	}

}
