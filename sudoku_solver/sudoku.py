import numpy as np
import cv2


class SudokuImage:
    """A standard 9x9 Sudoku puzzle represented by an image. The grid can be extracted from the image.
    """

    def __init__(self, file: str) -> None:
        """Initialize the image.

        Args:
            file (str): Path to the image file.
        """
        self.img = self.resize_height(cv2.imread(file), 500)

    def show(self, solution: bool = True, border: bool = False) -> None:
        if border:
            cv2.drawContours(self.img, [self.get_border()], 0, (0, 255, 0), 1)

        if solution:
            # TODO: Display solution obtained by the solver.
            pass

        cv2.imshow("Sudoku", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_height(self, img: np.ndarray, height: int) -> np.ndarray:
        """Resize image to a given height but keep aspect ratio.

            Args:
                img (np.ndarray): The image to resize.
                height (int): The new height of the image.

            Returns:
                np.ndarray: The resized image.
            """
        original_width, original_height = img.shape[:2]
        scale_factor = height / original_height
        width = int(scale_factor * original_width)

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def get_border(self) -> np.ndarray:
        """Find the border of the Sudoku grid.

        Returns:
            np.ndarray: The contour of the border.
        """
        # Preprocess image to find the border easier
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)

        # Find the border by assuming the Sudoku grid is the 2nd largest area
        # TODO: Find a better way to extract the border since this crude assumption will not hold in most cases.
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        border = sorted(contours, key=cv2.contourArea, reverse=True)[1]

        return border

    def deskew(self) -> None:
        # TODO: Transform image to top-down view
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        # TODO: Extract numbers and represent the puzzle as a 9x9 numpy array
        raise NotImplementedError


class SudokuSolver:
    # TODO: Solve the puzzle given a 9x9 numpy array.
    pass


if __name__ == "__main__":
    sudoku = SudokuImage("img/sudoku1.jpg")
    sudoku.show(border=True)
