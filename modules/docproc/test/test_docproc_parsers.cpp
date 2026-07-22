// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/doctags_parser.hpp"
#include "../src/markdown_parser.hpp"
#include "../src/plain_text_parser.hpp"

namespace opencv_test { namespace {

using namespace cv::docproc;

TEST(Docproc_DocTagsParser, IsDocTags)
{
    EXPECT_TRUE(isDocTags("<doctag><text>hi</text></doctag>"));
    EXPECT_FALSE(isDocTags("plain text"));
}

TEST(Docproc_DocTagsParser, TextTagBecomesTextBlock)
{
    Page page;
    parseDocTagsPage("<doctag><text>Hello world</text></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Hello world", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, HeaderTagBecomesTitleBlock)
{
    Page page;
    parseDocTagsPage(
        "<doctag><section_header_level_1>Title Text</section_header_level_1></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title Text", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, RaggedTableColsMatchesWidestRow)
{
    Page page;
    parseDocTagsPage(
        "<doctag><otsl><fcel>A<fcel>B<nl><fcel>C<fcel>D<fcel>E<nl></otsl></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(3, block.cols); // widest row (3 cells), not the first row (2 cells)
    ASSERT_EQ((size_t)5, block.cells.size());
}

TEST(Docproc_DocTagsParser, IsOtslCellStream)
{
    EXPECT_TRUE(isOtslCellStream("<fcel>A<fcel>B<nl>"));
    EXPECT_FALSE(isOtslCellStream("<doctag><text>hi</text></doctag>"));
}

TEST(Docproc_DocTagsParser, ParseOtslCellStreamHasNoClosingTag)
{
    Page page;
    parseOtslCellStream("<fcel>A<fcel>B<nl>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(1, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)2, block.cells.size());
    EXPECT_EQ("A", block.cells[0].text);
    EXPECT_EQ("B", block.cells[1].text);
}

TEST(Docproc_MarkdownParser, IsMarkdown)
{
    EXPECT_TRUE(isMarkdown("# Title\n"));
    EXPECT_TRUE(isMarkdown("| A | B |\n| --- | --- |\n| 1 | 2 |\n"));
    EXPECT_FALSE(isMarkdown("Just plain text.\nAnother line.\n"));
}

TEST(Docproc_MarkdownParser, HeaderAndParagraph)
{
    Page page;
    parseMarkdownPage("# Title\n\nSome paragraph text.", page);

    ASSERT_EQ((size_t)2, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title", page.blocks[0].lines[0].text);
    EXPECT_EQ(BLOCK_TEXT, page.blocks[1].type);
    EXPECT_EQ("Some paragraph text.", page.blocks[1].lines[0].text);
}

TEST(Docproc_MarkdownParser, PipeTable)
{
    Page page;
    parseMarkdownPage("| A | B |\n| --- | --- |\n| 1 | 2 |", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)4, block.cells.size());
    EXPECT_TRUE(block.cells[0].is_header);
    EXPECT_FALSE(block.cells[2].is_header);
    EXPECT_EQ("1", block.cells[2].text);
}

TEST(Docproc_PlainTextParser, SingleSpaceStaysInOneColumn)
{
    Page page;
    parsePlainTextPage("Jane Miller works here.", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Jane Miller works here.", page.blocks[0].lines[0].text);
}

TEST(Docproc_PlainTextParser, WhitespaceAlignedColumnsBecomeTable)
{
    Page page;
    parsePlainTextPage("Name         Age\nJane Miller  30\nJohn Smith   25", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(3, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)6, block.cells.size());
    EXPECT_EQ("Jane Miller", block.cells[2].text); // row 1, col 0 -- not split on the single space
    EXPECT_FALSE(block.cells[2].is_header);
}

}} // namespace opencv_test::(anonymous)
