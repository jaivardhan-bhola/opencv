// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DOCPROC_DOCTAGS_PARSER_HPP
#define OPENCV_DOCPROC_DOCTAGS_PARSER_HPP

#include "opencv2/docproc.hpp"

namespace cv { namespace docproc {

/** @brief True if raw looks like DocTags-style markup (e.g. Granite-Docling's output),
as opposed to plain OCR text (e.g. PaddleOCR-VL's default output). */
bool isDocTags(const String& raw);

/** @brief Parses DocTags-style markup into page.blocks.

DocTags location tags (`<loc_N>`) are consumed but not converted into bbox values;
Block/Line/Cell::bbox are left at their default (empty) value.
*/
void parseDocTagsPage(const String& raw, Page& page);

/** @brief True if raw is a bare OTSL cell stream (`<fcel>`/`<ched>`/`<ecel>`/`<lcel>`/`<nl>`)
with no `<doctag>`/`<otsl>` wrapper, as produced by PaddleOCR-VL's "Table Recognition:" prompt. */
bool isOtslCellStream(const String& raw);

/** @brief Parses a bare OTSL cell stream into a single BLOCK_TABLE page.blocks entry. */
void parseOtslCellStream(const String& raw, Page& page);

}} // namespace cv::docproc

#endif // OPENCV_DOCPROC_DOCTAGS_PARSER_HPP
